import torch
from torch import nn, distributions
from torch.distributions import categorical
from torch.nn import functional as F
from itertools import chain, combinations, product
from functools import lru_cache
from im_net import activation_functions as af
from im_net import helper_functions as hf
from im_net import prob_estim

@lru_cache
def load_moebius_function(n, device="cpu"):
    import pickle as pkl

    moebius_file_name = "moebius.pkl"
    with open(moebius_file_name, "rb") as moebius_file:
        moebius_data = pkl.load(moebius_file)
    inversion_matrix = moebius_data[n][1]
    antichains = moebius_data[n][0]
    return torch.tensor(inversion_matrix, device=device), antichains

class IM_Layer(nn.Module):
    def __init__(
        self,
        input_sizes,
        output_size,
        activation,
        binning,
        biases=[True, True],
        connections=[1, 1],
        discrete_output_values=[-1, 1],
        activation_params={},
    ):
        super().__init__()
        self.binning = binning
        self.input_sizes = input_sizes
        self.output_size = output_size
        self.discrete_output_values = torch.tensor(
            discrete_output_values, dtype=torch.get_default_dtype()
        )
        self.sources = nn.ModuleList(
            nn.Linear(int(input_size), output_size, bias=biases[i])
            for i, input_size in enumerate(input_sizes)
        )
        self.activation = hf.load_module(activation)(
            output_size=output_size, **activation_params
        )
        self.init_connections(connections)
        self.save_for_loss = None

    def create_im_layer(self, layer_params, bin_methods):
        """Helper method to create an IM_Layer."""
        activation_type = layer_params.activation.type
        if self.global_opt:
            activation_type = 'im_net.activation_functions.Schneider3SourceActivationHeaviside'
        return im_model.IM_Layer(
            layer_params.input_sizes,
            layer_params.output_size,
            activation_type, 
            bin_methods[layer_params.binning],
            connections=[1, 1, 1-torch.eye(layer_params.output_size)], 
            biases=layer_params.bias,
            discrete_output_values=layer_params.discrete_output_values, 
            activation_params=layer_params.activation.params
        )

    def init_connections(self, connections):
        """Initializes the connections between the input and output neurons based on the connections parameter."""
        def get_grad_filter(fltr):
            # used for fixing the weights that are not connected (always 0)
            def backhook(grad):
                grad = grad * fltr.to(grad.device)
                return grad
            return backhook

        for i, c in enumerate(connections):
            if type(c) == int and c == 1:
                w_con = torch.ones(self.output_size, self.input_sizes[i])
            elif type(c) == int and c == 0:
                w_con = torch.zeros(self.output_size, self.input_sizes[i])
            else:
                assert c.shape == (self.output_size, self.input_sizes[i])
                w_con = c
            self.sources[i].weight = nn.Parameter(
                self.sources[i].weight * w_con
            )
            self.sources[i].weight.register_hook(get_grad_filter(w_con))

    def loss(
        self,
        gamma=torch.Tensor([0.1, 0.1, 1, 0.1, 0.1]),
        return_information=False,
        return_information_tensor=False,
    ):
        x, thetas = self.save_for_loss

        if len(self.input_sizes) == 2: # Bivariate neurons
            permutation = [2, 3, 1, 0]
        else:                        # Trivariate neurons
            permutation = [2, 3, 4, 1, 0]

        all_theta= self.binning.get_theta_hist(thetas, x, self.output_size) / x[0].shape[0]
        pid_results = self.pid(
            all_theta.permute(*permutation)
        )  #  reshape to [r, c,(l,) theta, neuron]

        loss = -torch.tensor(gamma, dtype=torch.get_default_dtype(), device=pid_results.device) @ pid_results.sum(-1).type(torch.get_default_dtype())
        if return_information_tensor:
            return loss, pid_results
                    
        if return_information:
            return loss, pid_results.detach().cpu().numpy()

        return loss

    @staticmethod
    def powerset(iterable):
        "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    
    def compute_marginals(self, n_sources, p_all_neurons):

        p = dict()
        for a in self.powerset(range(n_sources + 1)):
            if len(a) == n_sources + 1:
                p[a] = p_all_neurons
                continue

            p[a] = p_all_neurons.sum(tuple(set(range(n_sources + 1)) - set(a)), keepdim=True)
        return p
    
    def union_probability(self, n_sources, p_marginal, alpha):
        """ Computes the probability of the union of all sets in alpha
        using the inclusion-exclusion principle """
        res = torch.zeros_like(p_marginal[tuple(range(n_sources + 1,))])
        for k in range(1, len(alpha) + 1):
            for beta in combinations(alpha, k):
                b_union = tuple(set(b for a in beta for b in a))
                res += (-1) ** (k - 1) * p_marginal[b_union]
        return res
    
    def compute_isx(self, n_sources, p_marginal, alpha):

        with torch.no_grad():
            p_union = self.union_probability(n_sources, p_marginal, alpha)

        alpha_t = tuple(a + (n_sources,) for a in alpha)

        p_union_t = self.union_probability(n_sources, p_marginal, alpha_t)

        frac = p_union_t / (p_union * p_marginal[(n_sources,)] + 1e-10)

        # Avoid NaN Gradients by filtering out small values
        frac[frac < 1e-7] = 1

        return (p_marginal[tuple(range(n_sources+1))] * torch.log2(frac)).sum(tuple(range(n_sources+1)))
    

    def pid(self, p_all_neurons):

        n_sources = p_all_neurons.dim() - 2

        moebius, antichains = load_moebius_function(n_sources, device=p_all_neurons.device)
        moebius = moebius.to(p_all_neurons.dtype)
        antichains = [tuple(tuple(i-1 for i in a) for a in antichain) for antichain in antichains] # Convert to zero-based

        # Compute all marginals:
        # size is (20*20*20*2 + 20*20*20 + 3 * 20*20*2 + 3 * 20*20 + 3 * 20*2 + 3 * 20 + 2 + 1) * 1000 * 64bit = 1.7GB
        p_marginal = self.compute_marginals(n_sources, p_all_neurons)

        # Compute all I_cap terms:
        I_cap = torch.zeros((len(antichains), p_all_neurons.shape[-1]), device=p_all_neurons.device)
        for i, alpha in enumerate(antichains):
            I_cap[i] = self.compute_isx(n_sources, p_marginal, alpha)

        # Compute atoms
        Pi = moebius @ I_cap

        # Compute residual entropy H_res
        H_tot = -torch.sum(p_marginal[(n_sources,)] * (torch.log2(p_marginal[(n_sources,)]+1e-10)), -2)
        H_res = H_tot - I_cap[antichains.index((tuple(range(n_sources)),))]

        # Convert to the expected order
        antichains_ordered = hf.get_achains(num_sources=n_sources, starting_idx=0)
        info_terms = torch.empty((Pi.shape[0]+1, *Pi.shape[1:]), device=p_all_neurons.device, dtype=p_all_neurons.dtype)
        for i, antichain in enumerate(antichains_ordered):
            info_terms[i] = Pi[antichains.index(antichain)]
        info_terms[-1] = H_res

        return info_terms
        
    def forward(self, x, sample=True, use_max=False):

        val = [None] * len(self.sources)
        for i, sources in enumerate(self.sources):
            val[i] = sources(x[i].to(x[0].device))
            
        if self.binning.use_bincenters:
            val = self.binning.center_ND(val)

        if self.activation.has_surrogate:
            out = self.activation(*val)
            out_probs = torch.stack([out, 1 - out], axis=2)
            self.save_for_loss = (val, out_probs)
            return out
        
        out_probs = self.activation(*val)
        self.save_for_loss = (val, out_probs)

        if sample:
            with torch.no_grad():
                out_probs[out_probs < 0] = 0
                if torch.any(torch.isnan(out_probs)):
                    out_probs = out_probs.nan_to_num(0, 0, 0)            
            if use_max:
                output = out_probs.argmax(axis=-1)
                return self.discrete_output_values.to(x[0].device)[output]
            try:
                output = categorical.Categorical(out_probs, validate_args=False).sample()
            except ValueError:
                print(out_probs)
                raise ValueError("Thetas are no simplex")
            return self.discrete_output_values.to(x[0].device)[output]
        return out_probs
