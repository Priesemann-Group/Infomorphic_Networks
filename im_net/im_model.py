import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / "external" / "dit"))
sys.path.append(str(Path(__file__).parents[1] / "external" / "BROJA_2PID"))
import torch
from torch import nn
from torch.distributions import categorical
import numpy as np
import tqdm
from itertools import chain, combinations, product
from functools import lru_cache
from im_net import helper_functions as hf
from im_net import activation_functions as af
from im_net import prob_estim

@lru_cache
def load_reordered_moebius_function(n, device="cpu"):
    import pickle as pkl

    moebius_file_name = "moebius.pkl"
    with open(moebius_file_name, "rb") as moebius_file:
        moebius_data = pkl.load(moebius_file)
    antichains, inversion_matrix = moebius_data[n]
    return torch.as_tensor(inversion_matrix, device=device, dtype=torch.get_default_dtype()), antichains

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
        self.register_buffer(
            "discrete_output_values",
            torch.tensor(discrete_output_values, dtype=torch.get_default_dtype()),
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

    def init_connections(self, connections):
        """Initialise per-source connectivity masks.

        Masks are stored as buffers (so they follow the module to GPU) and a
        backward hook zeroes the gradient on masked-out weights. Fully-connected
        sources (``c == 1``) need no mask and no hook — skip them entirely.
        """
        for i, c in enumerate(connections):
            if isinstance(c, int):
                assert c in (0, 1), f"connection int must be 0 or 1, got {c}"
                if c == 1:
                    # Fully connected — no masking needed
                    continue
                mask = torch.zeros(self.output_size, self.input_sizes[i])
            else:
                assert c.shape == (self.output_size, self.input_sizes[i])
                mask = c
            buf_name = f"_conn_mask_{i}"
            self.register_buffer(buf_name, mask)
            # Apply the mask to the initial weights
            self.sources[i].weight = nn.Parameter(self.sources[i].weight * mask)
            # Hook looks the buffer up through ``self`` so it always sees the
            # current device after ``.to(...)``/``.cuda()``.
            self.sources[i].weight.register_hook(self._grad_mask_hook(buf_name))

    def _grad_mask_hook(self, buf_name):
        """Backward-hook factory that masks gradients using the named buffer."""
        def backhook(grad):
            return grad * getattr(self, buf_name)
        return backhook

    def loss(
        self,
        gamma=torch.Tensor([0.1, 0.1, 1, 0.1, 0.1]),
        return_atoms=False,
            ):
        
        x, thetas = self.save_for_loss
        if not isinstance(gamma,torch.Tensor):
            gamma = torch.tensor(gamma,dtype=torch.get_default_dtype(),device=x[0].device)
        
        full_distribution = self.binning.get_theta_hist(thetas, x) / x[0].shape[0]

        permutation = list(range(2, 2+len(self.input_sizes))) + [1, 0]

        pid_results = InformationDecomposition(
            full_distribution.permute(*permutation)
        )

        loss = -gamma @ pid_results.sum(-1).type(torch.get_default_dtype())

        if return_atoms:
            return loss, pid_results

        return loss

    def dit_pids(self, additional_measures=[]):
        if len(additional_measures) == 0:
            return 
        import dit
        x, thetas = self.save_for_loss
        additional_pids = {}
        # [0: neurons, 1: output, 2..: sources]  ->  [neurons, sources..., output]
        permutation = [0] + list(range(2, 2 + len(self.input_sizes))) + [1]

        all_theta = self.binning.get_theta_hist(thetas, x) / x[0].shape[0]

        # Wrapper for DIT PID computation
        # first, transform p_all_neurons to the shape expected by DIT

        def tensor_to_dit_dist(tensor):
            data_np = tensor.detach().cpu().numpy()

            n_neurons, s_1_dim, s_2_dim, t_dim = data_np.shape
            outcomes = list(product(range(s_1_dim), range(s_2_dim), range(t_dim)))

            distributions = []

            for neuron_idx in tqdm.tqdm(range(n_neurons)):
                pmf_flat = data_np[neuron_idx].flatten()
                pmf_flat = pmf_flat / pmf_flat.sum()

                d = dit.Distribution(outcomes, pmf_flat)
                distributions.append(d)
            return distributions
        
        def tensor_to_dict(tensor):
            data_np = tensor.detach().cpu().numpy()

            n_neurons, t_dim, s_1_dim, s_2_dim = data_np.shape
            outcomes = list(product(range(t_dim), range(s_1_dim), range(s_2_dim)))

            distributions = []
            for neuron_idx in range(n_neurons):
                dist_dict = {}
                pmf_flat = np.float64(data_np[neuron_idx].flatten())
                pmf_flat = pmf_flat / pmf_flat.sum()
                
                for outcome, prob in zip(outcomes, pmf_flat):
                    dist_dict[outcome] = float(prob)
                distributions.append(dist_dict)
            return distributions

        
        dit_distributions = tensor_to_dit_dist(all_theta.permute(*permutation)) 
        n_neurons = len(dit_distributions)
        antichains = [((0,),),((1,),),((0,), (1,)),((0, 1),)]

        def _compute_dit_pid(pid_class, name):
            result = np.zeros((len(antichains), len(dit_distributions)))
            for i, dist in tqdm.tqdm(enumerate(dit_distributions), desc=f"Computing {name} PID"):
                pid_result = pid_class(dist, sources=[(0,), (1,)], target=(2,))
                result[:, i] = [pid_result[ac] for ac in antichains]
            return result

        if {'SX', 'WB', 'PM', 'MMI'} & set(additional_measures):
            import dit.pid as _dit_pid
            for measure in ('SX', 'WB', 'PM', 'MMI'):
                if measure in additional_measures:
                    additional_pids[measure] = _compute_dit_pid(
                        getattr(_dit_pid, f'PID_{measure}'), measure
                    )

        if 'BROJA' in additional_measures:
            additional_pids['BROJA'] = np.zeros((len(antichains), len(dit_distributions)))
            atom_names = ['UIY', 'UIZ', 'SI', 'CI']
            from broja2pid import BROJA_2PID  # using BROJA_2PID as it is faster
            dist_dicts = tensor_to_dict(all_theta)
            for i, dist in tqdm.tqdm(enumerate(dist_dicts), desc="Computing BROJA PID"):
                pid_result = BROJA_2PID.pid(dist, output=-1)
                additional_pids['BROJA'][:, i] = [pid_result[an] for an in atom_names]
        return additional_pids

    def forward(self, x, sample=True, use_max=False, return_probs=False, save_for_loss=True):
        val = [source(x[i].to(x[0].device)) for i, source in enumerate(self.sources)]

        # Deterministic / surrogate-gradient activations return a single tensor
        # of "spike" probabilities; wrap into a 2-column simplex for save_for_loss.
        is_binary_heaviside = self.activation._get_name() == "BinaryHeaviside"
        if is_binary_heaviside or self.activation.has_surrogate:
            out = self.activation(*val)
            out_probs = torch.stack([out, 1 - out], axis=2)
            if save_for_loss:
                self.save_for_loss = (val, out_probs)
            if is_binary_heaviside:
                return self.discrete_output_values[out.int()]
            return out

        out_probs = self.activation(*val)
        if save_for_loss and self.training:
            self.save_for_loss = (val, out_probs)

        if not sample:
            return out_probs

        with torch.no_grad():
            out_probs[out_probs < 0] = 0
            if torch.any(torch.isnan(out_probs)):
                out_probs = out_probs.nan_to_num(0, 0, 0)

        if use_max:
            output = out_probs.argmax(axis=-1)
            return self.discrete_output_values[output]

        try:
            output = categorical.Categorical(out_probs, validate_args=False).sample()
        except ValueError:
            print(out_probs)
            raise ValueError("Thetas are no simplex")

        if return_probs:
            return self.discrete_output_values[output].detach(), out_probs[:,:,0]
        return self.discrete_output_values[output]
    

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def compute_marginals(n_sources, p_all_neurons):
    p = dict()
    for a in powerset(range(n_sources + 1)):
        if len(a) == n_sources + 1:
            p[a] = p_all_neurons
            continue

        p[a] = p_all_neurons.sum(tuple(set(range(n_sources + 1)) - set(a)), keepdim=True)
    return p


def union_probability(n_sources, p_marginal, alpha):
    """ Computes the probability of the union of all sets in alpha
    using the inclusion-exclusion principle """
    res = torch.zeros_like(p_marginal[tuple(range(n_sources + 1,))])
    for k in range(1, len(alpha) + 1):
        for beta in combinations(alpha, k):
            b_union = tuple(set(b for a in beta for b in a))
            res += (-1) ** (k - 1) * p_marginal[b_union]
    return res


def compute_isx(n_sources, p_marginal, alpha):
    with torch.no_grad():
        p_union = union_probability(n_sources, p_marginal, alpha)

    alpha_t = tuple(a + (n_sources,) for a in alpha)

    p_union_t = union_probability(n_sources, p_marginal, alpha_t)

    frac = p_union_t / (p_union * p_marginal[(n_sources,)] + 1e-10)

    # Avoid NaN Gradients by filtering out small values
    frac[frac < 1e-7] = 1

    return (p_marginal[tuple(range(n_sources+1))] * torch.log2(frac)).sum(tuple(range(n_sources+1)))


def InformationDecomposition(p_all_neurons):
    n_sources = p_all_neurons.dim() - 2

    moebius, antichains = load_reordered_moebius_function(n_sources, device=p_all_neurons.device)

    # Compute all marginals:
    p_marginal = compute_marginals(n_sources, p_all_neurons)

    # Compute all I_sx terms (antichains are pre-reordered, zero-based):
    I_sx = torch.zeros((len(antichains), p_all_neurons.shape[-1]), device=p_all_neurons.device)
    for i, alpha in enumerate(antichains):
        I_sx[i] = compute_isx(n_sources, p_marginal, alpha)

    # Compute atoms via Moebius inversion
    atoms = moebius @ I_sx

    # Append residual entropy: H_res = H_tot - I({0,...,n-1}; T), and the
    # last antichain in the reordered list is (tuple(range(n_sources)),)
    total_entropy = -torch.sum(p_marginal[(n_sources,)] * (torch.log2(p_marginal[(n_sources,)]+1e-10)), -2)
    residual = (total_entropy - I_sx[-1]).reshape(1, -1)

    return torch.cat((atoms, residual), dim=0)


def create_im_layer(self, layer_params, bin_methods):
    """Helper method to create an IM_Layer."""
    activation_type = layer_params.activation.type
    if self.global_opt:
        activation_type = 'im_net.activation_functions.Schneider3SourceActivationHeaviside'
    return IM_Layer(
        layer_params.input_sizes,
        layer_params.output_size,
        activation_type, 
        bin_methods[layer_params.binning],
        connections=[1, 1, 1-torch.eye(layer_params.output_size)], 
        biases=layer_params.bias,
        discrete_output_values=layer_params.discrete_output_values, 
        activation_params=layer_params.activation.params
    )

