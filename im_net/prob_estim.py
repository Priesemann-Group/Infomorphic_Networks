from functools import reduce
from operator import mul
import warnings
import torch
import numpy as np
import im_net.helper_functions as hf

class HeavisideSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).type(torch.get_default_dtype()) # this is potentially unfair, since our networks work with probabilities to get the accuracy

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
        return sigmoid_grad * grad_output, None, None
    
def init_prob_estim_methods(binning_params, device):
    bin_methods = dict()
    for binning in binning_params:
        bin_methods[binning] = hf.load_module(binning_params[binning].name)(
            device, **binning_params[binning].params
        )
    return bin_methods


class BaseBinning(): # Base class for binning methods which is never used directly
    """Interface for all binning methods. Contains all functions that are used by multiple binning methods.
    Attributes:
        device: Torch device.
        n_bins: Number of bins. List[N_sources]
        edges: optional, The upper and lower limits of the binning range. List[(lower,upper),...]
        normalize: If True, the input variables are normalized to the range [-1,1] before binning.
    """
    def __init__(self, device, n_bins, edges=None, normalize=False, padding=0):
        self.n_bins = n_bins
        self.device = device
        if edges is not None:
            self.edges = [torch.tensor(e, device=device) for e in edges]
            self.binedges = [
                torch.linspace(*e, n_bins[i] + 1, device=device)
                for i, e in enumerate(self.edges)
            ]
        else:
            self.edges = None
            self.binedges = None
        
        self.normalize = normalize
        self.padding = padding  # Padding for the binning edges, e.g. to avoid numerical issues with torch.bucketize
    
    def get_theta_hist(self, thetas:torch.Tensor, x:list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            thetas: Firing probabilities. Shape: (m,n,o) where m is the amount of samples, n the neurons and o the amount of output types.
            x: Input variables. Shape: List [(m,n),...] with an entry for each input source.
        Returns:
            The binned total counts. Shape: (n,o,*n_bins) where n are the neurons and o the output types.
        """
        bin_product = reduce(mul, self.n_bins, 1)
        batch_size, n_neurons, o = thetas.shape

        with torch.no_grad():
            bins = self.get_buckets(x)

            # Combine bins together by creating a unique index for each multi-dimensional bin
            combined_bins = bins[0]
            for i in range(1, len(bins)):
                combined_bins *= self.n_bins[i]
                combined_bins += bins[i]
            index = combined_bins[..., None].expand(-1, n_neurons, o)

        # similar to torch.histogram(index, weight=thetas), but with gradients for the weights
        result = torch.zeros(bin_product, n_neurons, o, device=self.device, dtype=thetas.dtype)
        result.scatter_add_(0, index, thetas)
        result = result.view(*tuple(self.n_bins), n_neurons, o).movedim((-2, -1), (0, 1)).contiguous()
        return result

    def get_bin_centers(self):
        bincenters = [None] * len(self.edges)
        for i, e in enumerate(self.edges):
            bincenters[i] = self.binedges[i][:-1] + (e[1] - e[0]).abs().sum() / (
                2 * (self.n_bins[i] - 1)
            )
        return tuple(bincenters)
        
    def get_buckets(self, x):
        return NotImplementedError
    
    def normalize_input(self, x):
        x_normalized = []
        for tensor in x:
            tensor_max = tensor.abs().max(dim=0, keepdim=True)[0]
            normalized_tensor = tensor / tensor_max
            x_normalized.append(normalized_tensor)
        return x_normalized
    
    def reset_edges(self, x):
        if self.normalize:
            self.edges = [torch.tensor([-1-self.padding, 1+self.padding], device=self.device) for _ in range(len(x))]
        else:
            self.edges = [torch.tensor([x[i].min()-self.padding, x[i].max()+self.padding], device=self.device) for i in range(len(x))]

        self.binedges = [
            torch.linspace(
                e[0], e[1], self.n_bins[i] + 1, device=self.device
            )
            for i, e in enumerate(self.edges)
        ]
    
class BinningFixedSize(BaseBinning):  # actually not fixed size, but fixed range
    def __init__(self, device, n_bins, edges=None, normalize=False):
        super().__init__(device, n_bins, edges, normalize)

    def get_buckets(self, x):
        if self.normalize:
            x = self.normalize_input(x)
        buckets = tuple(
            torch.bucketize(
                torch.clamp(x[i], self.binedges[i][0] + 1e-4, self.binedges[i][-1]),
                self.binedges[i]
            ) - 1
            for i in range(len(x))
        )
        return buckets

class BinningAdaptiveSize(BaseBinning):
    def __init__(self, device, n_bins, edges=None, normalize=False, padding=0):
        super().__init__(device, n_bins, edges, normalize, padding)

    def get_buckets(self, x):
        if self.normalize:
            x = self.normalize_input(x)
        self.reset_edges(x)
        buckets = tuple(
            torch.bucketize(
                torch.clamp(x[i], self.binedges[i][0] + 1e-4, self.binedges[i][-1]),
                self.binedges[i]
            ) - 1
            for i in range(len(x))
        )
        return buckets

class BinningCDFDiff(BaseBinning):
    """
    Adaptive N-D histogram of firing probabilities via logistic-CDF differences.
    Provides both get_theta_hist (mass-weighted histogram) and get_hist (plain histogram)
    with optional absolute or relative smoothing widths.

    Args:
        device
        n_bins         : list of ints, number of bins per axis
        width_factor   : float, larger → sharper bins (2–4 is typical)
        absolute_width : optional float or list of floats, if provided overrides relative width
    """
    def __init__(self, device, n_bins, edges, width_factor=2.5, absolute_width=None,
                 normalize=False, padding=0):
        super().__init__(device, n_bins, edges,
                         normalize=normalize, padding=padding)
        self.width_factor = width_factor
        self.absolute_width = absolute_width  # scalar or list/tuple of length D

    def get_theta_hist(self,
                       thetas: torch.Tensor,
                       x: list[torch.Tensor],
                       ) -> torch.Tensor:
        """
        Compute mass-weighted N-D histogram.

        Args:
            thetas: Tensor of shape (m, n, o) — m samples, n neurons, o output-types.
            x     : list of n_sources Tensors, each of shape (m, n) for each input dimension.
        Returns:
            hist: Tensor of shape (n, o, *n_bins), giving for each neuron & output-type the histogram.
        """
        if self.normalize:
            x = self.normalize_input(x)
        self.reset_edges(x)
        batch_size, n_neurons, o = thetas.shape
        n_sources = len(x)
        W_prod = None

        for d, xd in enumerate(x):
            edges = self.binedges[d]                   # (n_d+1,)
            extended = torch.cat([
                torch.full((1,), -float('inf'), device=edges.device),
                edges[1:-1],
                torch.full((1,),  float('inf'), device=edges.device),
            ], dim=0)
            lo, hi = extended[:-1], extended[1:]       # each (n_d,)

            # determine smoothing width for axis d
            if self.absolute_width is not None:
                W_val = (self.absolute_width[d] if isinstance(self.absolute_width, (list, tuple))
                         else self.absolute_width)
                W = torch.tensor(W_val, device=edges.device)
            else:
                bin_width = (edges[1] - edges[0]).mean()
                W = bin_width / self.width_factor

            z_hi = (hi[None, None, :] - xd.unsqueeze(-1)) / W
            z_lo = (lo[None, None, :] - xd.unsqueeze(-1)) / W
            w_d = torch.sigmoid(z_hi) - torch.sigmoid(z_lo)  # (m, n, n_d)

            # reshape for separable outer-product
            shape = [batch_size, n_neurons] + [1] * n_sources
            shape[2 + d] = w_d.size(-1)
            w_d = w_d.view(*shape)

            W_prod = w_d if W_prod is None else W_prod * w_d

        # weight by thetas and sum over samples
        thetas_ = thetas.view(batch_size, n_neurons, *([1] * n_sources), o)
        weighted = W_prod.unsqueeze(-1).detach() * thetas_  # (m, n, *bins, o)
        H = weighted.sum(dim=0)  # (n, *bins, o)

        # reorder to (n, o, *bins)
        perm = [0, n_sources + 1] + list(range(1, n_sources + 1))
        return H.permute(*perm).contiguous()

    def get_hist(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute plain N-D histogram of counts.

        Args:
            x: list of D Tensors, each of shape (m,) for each input dimension.
        Returns:
            hist: Tensor of shape (*n_bins), the histogram counts over all samples.
        """
        x_wrapped = [xd.unsqueeze(-1) for xd in x]
        m = x[0].shape[0]
        thetas = torch.ones(m, 1, 1, device=x[0].device)
        hist = self.get_theta_hist(thetas, x_wrapped)
        return hist.squeeze(0).squeeze(0)
    
class BinningDifferentiable(BaseBinning):
    """
    Differentiable approximation to the binning process. Uses a Gaussian kernel to allocate weights to bins.
    The parameter sigma controls the sharpness of the kernel.
    """
    def __init__(self, device, n_bins, edges=None,normalize=False,
                adaptive_sigma=False,sigma_fraction=None,sigma=1, kernel='Gaussian', 
                 padding=0):
        super().__init__(device, n_bins, edges, normalize, padding)
        self.sigma = sigma
        self.adaptive_sigma = adaptive_sigma
        self.sigma_fraction = sigma_fraction
        self.kernel = kernel

    def get_theta_hist(self, thetas, x, sigma=None, kernel=None):
        if self.normalize:
            x = self.normalize_input(x)
        self.reset_edges(x)
        batch_size, n_neurons, o = thetas.shape

        if sigma is not None:
            self.sigma = sigma
        # elif self.sigma is None:
        #     bin_widths = [(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)]
        #     self.sigma = (self.get_smallest_sigma()+max(bin_widths))/2
        
        if kernel is not None:
            self.kernel = kernel

        self.reset_edges(x)
        if self.adaptive_sigma:
           self.reset_sigma()

        self.check_sigma()
        bin_widths = [(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)]
        # self.sigma = (self.get_smallest_sigma()+max(bin_widths))/2
        bin_product = reduce(mul, self.n_bins, 1)
        x = torch.stack(x, dim=-1)

        bin_centers = self.get_bin_centers()
        bin_center_grid = torch.cartesian_prod(*bin_centers)

        weights = torch.cdist(x, bin_center_grid, p=2).view(batch_size, n_neurons, bin_product)
        weights = self.kernels(weights, self.sigma, self.kernel)

        #nan to num should not be necessary, but sometimes the weights are nan if sigma is too small!
        weights = torch.nan_to_num(weights / (weights.sum(dim=-1, keepdim=True)), nan=0)
        if weights.isnan().any():
            raise ValueError('Some weights are Nan.')

        conditional_samples = weights.unsqueeze(-1) * thetas.unsqueeze(2)
        #sum over samples
        conditional_prob_mass = conditional_samples.mean(0)
        #recast result to grid
        result = conditional_prob_mass.transpose(1,2).reshape(n_neurons,o,*self.n_bins).contiguous()
        result = (batch_size)*result  #back to histogram normalization
        return result
    
    def get_hist(self, x, kernel='Gaussian'):
        # self.reset_edges(x)
        batch_size = x[0].shape[0]
        bin_product = reduce(mul, self.n_bins, 1)
        x = torch.stack(x, dim=-1)
        bin_widths = [(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)]
        self.sigma = max(bin_widths)
        bin_centers = self.get_bin_centers()
        bin_center_grid = torch.cartesian_prod(*bin_centers)

        weights = torch.cdist(x, bin_center_grid, p=2).view(batch_size, bin_product)
        weights = self.kernels(weights, self.sigma, self.kernel)

        dims = tuple(i for i in range(x.ndim) if i != 0)
        weights = torch.nan_to_num(weights / (weights.sum(dim=dims, keepdim=True)), nan=0)

        weights = weights.mean(dimGaussian=0)
        weights = weights.view(*tuple(self.n_bins)).contiguous()
        return weights

    def get_bin_centers(self):
        bincenters = [None] * len(self.edges)
        for i, e in enumerate(self.edges):
            bincenters[i] = self.binedges[i][:-1]+(e[1]-e[0]).abs().sum() / (2*self.n_bins[i])
        return tuple(bincenters)
    
    def reset_sigma(self):
        d_bins = [(edges[1]-edges[0])/n for edges,n in zip(self.edges,self.n_bins)]
        smallest_distance = min(d_bins)
        self.sigma = self.sigma_fraction * smallest_distance
    
    def check_sigma(self):
        bin_widths = torch.tensor([(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)])
        max_within_distance = torch.sqrt(torch.sum((bin_widths/2)**2 ))
        if self.kernels(max_within_distance, self.sigma, self.kernel) < 1e-38:
            warnings.warn('Warning: Sigma too small for bin width. Consider increasing!')
        # if self.sigma > min(bin_widths):
        #     warnings.warn('Warning: Sigma is larger than smallest bin width. Consider decreasing!')

    def get_smallest_sigma(self):
        bin_widths = torch.tensor([(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)])
        max_within_distance = torch.sqrt(torch.sum((bin_widths/2)**2 ))
        if self.kernel == 'Gaussian':
            return max_within_distance/torch.sqrt((-2*torch.log(torch.tensor(1e-38))))
        else:
            return max_within_distance

    def kernels(self, x, sigma, kernel='Gaussian'):
        if self.kernel == 'Gaussian':
            return torch.exp(-0.5*(x/sigma)**2)
        elif self.kernel == 'Epanechnikov':
            x = torch.tensor(1 - (x/sigma)**2)
            return torch.max(x,torch.tensor(0))
        elif self.kernel == 'Triangular':
            x = torch.tensor(1 - (x/sigma))
            return torch.max(x,torch.tensor(0))
        else:
            raise NotImplementedError


class BinningDifferentiableMarginal(BaseBinning):
    """
    Differentiable approximation to the binning process. Uses a Gaussian kernel to allocate weights to bins. Different than BinningDifferentiable, this acts on the marginal distributions of the input variables.
    The parameter sigma controls the sharpness of the kernel.
    """
    def __init__(self, device, n_bins, edges=None, normalize=False, sigma_fraction=None, sigma=1, kernel='Sigmoid', adaptive=True, padding=0):
        super().__init__(device, n_bins, edges, normalize, padding=padding)
        
        self.sigma = sigma
        self.sigma_fraction = sigma_fraction
        self.kernel = kernel
        self.adaptive = adaptive
        if edges is None and not adaptive:
            raise ValueError('Either edges must be provided or adaptive must be True.')

    def get_theta_hist(self, thetas, x, sigma=None, kernel='Sigmoid'):
        if self.normalize:
            x = self.normalize_input(x)
        if self.adaptive:
            self.reset_edges(x)
        x = torch.stack(x, dim=0)

        batch_size, n_neurons, o = thetas.shape

        if sigma is not None:
            self.sigma = sigma
        elif self.sigma is None:
            bin_widths = [(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)]
            self.sigma = bin_widths/2

        if kernel is not None:
            self.kernel = kernel

        bin_widths = [(e[1] - e[0]) / self.n_bins[i] for i, e in enumerate(self.edges)]
        
        if self.sigma_fraction is not None:
            if isinstance(self.sigma_fraction, (int, float)):
                self.sigma = [self.sigma_fraction * bin_widths[i] for i in range(len(bin_widths))]
            #can also be 'omegaconf.listconfig.ListConfig'
            else:
                self.sigma = [self.sigma_fraction[i] * bin_widths[i] for i in range(len(bin_widths))]
            # else:
            #     print(self.sigma_fraction)
            #     raise ValueError('sigma_fraction must be a float or a list of floats, not {}'.format(type(self.sigma_fraction)))
        

        bin_product = reduce(mul, self.n_bins, 1)

        bin_centers = self.get_bin_centers()
        
        weights = None
        if isinstance(self.sigma, (int, float)):
            self.sigma = [self.sigma] * len(x)
        for i, x_i in enumerate(x):
            width = (self.edges[i][1] - self.edges[i][0]) / self.n_bins[i]
            dists = torch.abs(x_i.unsqueeze(-1) - bin_centers[i])
            weight = self.kernels(dists, self.sigma[i]/width, width)
            shape = [batch_size, n_neurons] + [1] * len(x)  # [B, 1, 1, 1, ...]
            shape[i + 2] = -1  # place n_bins_i at correct position
            weight = weight.view(*shape)
            # weight = weight.unsqueeze(-i-1)
            weights = weight if weights is None else weights * weight
        
        weights = weights.view(batch_size, n_neurons, bin_product)
        weights = torch.nan_to_num(weights / (weights.sum(dim=-1, keepdim=True)), nan=0)

        conditional_samples = weights.unsqueeze(-1) * thetas.unsqueeze(2)
        
        #sum over samples
        conditional_prob_mass = conditional_samples.mean(0)
        #recast result to grid
        result = conditional_prob_mass.transpose(1,2).reshape(n_neurons,o,*self.n_bins).contiguous()
        result = (batch_size)*result  #back to histogram normalization
        return result
    
    def get_hist(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute plain N-D histogram of counts.

        Args:
            x: list of D Tensors, each of shape (m,) for each input dimension.
        Returns:
            hist: Tensor of shape (*n_bins), the histogram counts over all samples.
        """
        x_wrapped = [xd.unsqueeze(-1) for xd in x]
        m = x[0].shape[0]
        thetas = torch.ones(m, 1, 1, device=x[0].device)
        hist = self.get_theta_hist(thetas, x_wrapped)
        return hist.squeeze(0).squeeze(0), 0

    def get_bin_centers(self):
        bincenters = [None] * len(self.edges)
        for i, e in enumerate(self.edges):
            bincenters[i] = self.binedges[i][:-1]+(e[1]-e[0]).abs().sum() / (2*self.n_bins[i])
        return bincenters
    
    def reset_sigma(self):
        d_bins = [(edges[1]-edges[0])/n for edges,n in zip(self.edges,self.n_bins)]
        smallest_distance = min(d_bins)
        self.sigma = self.sigma_fraction * smallest_distance
    
    def kernels(self, x, sigma, width=None):
        if self.kernel == 'Gaussian':
            return torch.exp((-0.5*(x/sigma)**2))
        elif self.kernel == 'Epanechnikov':
            x = (1 - (x/sigma)**2)
            return torch.max(x,torch.tensor(0))
        elif self.kernel == 'Triangular':
            x = (1 - (x/sigma))
            return torch.max(x,torch.tensor(0))
        elif self.kernel == 'Sigmoid':
            return torch.sigmoid((width/2-x)/sigma)
        else:
            raise NotImplementedError


class BinningMaxEntropy(BaseBinning): # Pretty sure that this is wrong!!
    def __init__(self, device, n_bins, edges, normalize=False, padding=0):
        super().__init__(device, n_bins, edges,normalize, padding=padding)

    def reset_edges(self, x, normalize=False):
        for i, xs in enumerate(x):
            x_sorted, _ = torch.sort(xs.flatten(), dim=0)
            chunks = torch.chunk(x_sorted, self.n_bins[i], dim=0)
            self.n_bins[i] = len(chunks)
            self.binedges[i] = torch.zeros(self.n_bins[i], device=self.device)
            
            for j in range(self.n_bins[i] - 1):
                self.binedges[i][j+1] = (chunks[j][-1] + chunks[j + 1][0]) / 2
            if normalize:
                self.binedges[i][0] = 0
                self.binedges[i][-1] = 1
            else:
                self.binedges[i][0] = x_sorted[0] - 1
                self.binedges[i][-1] = x_sorted[-1] + 1

    def get_buckets(self, x):
        normalize=True
        buckets = [None] * len(x)
        if normalize:
            x_normalized = [(tensor - tensor.min(dim=0, keepdim=True)[0]) / (tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]) for tensor in x]
            x = x_normalized
        
        self.reset_edges(x, normalize=normalize)
        for i in range(len(x)):
            clamped = torch.clamp(
                x[i], self.binedges[i][0] + 1e-04, self.binedges[i][-1]
            )
            buckets[i] = torch.bucketize(clamped, self.binedges[i]) - 1
        return tuple(buckets)
