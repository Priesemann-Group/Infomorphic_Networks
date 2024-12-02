from functools import reduce
from operator import mul

import torch
import numpy as np
import matplotlib.pyplot as plt
import im_net.helper_functions as hf


def init_prob_estim_methods(binning_params, device):
    bin_methods = dict()
    for binning in binning_params:
        bin_methods[binning] = hf.load_module(binning_params[binning].name)(
            device, **binning_params[binning].params
        )
    return bin_methods


class BaseBinning(): # Base class for binning methods which is never used directly
    def __init__(self, device, n_bins, use_bincenters=False):
        self.n_bins = n_bins
        self.device = device
        self.use_bincenters = use_bincenters
    
    def get_theta_hist(self, thetas, x, n):

        batch_size = x[0].shape[0]
        bin_product = reduce(mul, self.n_bins, 1)

        with torch.no_grad():
            bins = self.get_buckets(x)

            # Combine bins together by creating a unique index for each multi-dimensional bin
            combined_bins = bins[0]
            for i in range(1, len(bins)):
                combined_bins *= self.n_bins[i]
                combined_bins += bins[i]

            index = combined_bins[..., None].expand(batch_size, n, thetas.shape[-1])

        # similar to torch.histogram(index, weight=thetas), but with gradients for the weights
        result = torch.zeros(bin_product, n, thetas.shape[-1], device=self.device, dtype=thetas.dtype)
        result.scatter_add_(0, index, thetas)
        result = result.view(*tuple(self.n_bins), n, thetas.shape[-1]).movedim((-2, -1), (0, 1)).contiguous()
        
        return result

    def get_bin_centers(self):
        bincenters = [None] * len(self.edges)
        for i, e in enumerate(self.edges):
            bincenters[i] = self.binedges[i][:-1] + (e[1] - e[0]).abs().sum() / (
                2 * (self.n_bins[i] - 1)
            )
        return tuple(bincenters)

    def center_ND(self, x):
        buckets = self.get_buckets(x)
        bincenters = self.get_bin_centers()
        centered = [None] * len(x)
        for i in range(len(x)):
            binned = bincenters[i][buckets[i]]
            delta = (x[i] - binned).detach()
            centered[i] = (x[i] - delta).float()
        return tuple(centered)

class BinningFixedSize(BaseBinning):  # actually not fixed size, but fixed range
    def __init__(self, device, n_bins, edges, use_bincenters=False, normalize=False, symmetric=False):
        super().__init__(device, n_bins, use_bincenters)
        self.edges = [torch.tensor(e, device=device) for e in edges]
        self.binedges = [
            torch.linspace(
                e[0], e[1], self.n_bins[i] + 1, dtype=torch.double, device=device
            )
            for i, e in enumerate(self.edges)
        ]


    def get_buckets(self, x):
        buckets = [None] * len(x)
        for i in range(len(x)):
            clamped = torch.clamp(
                x[i], self.binedges[i][0] + 1e-04, self.binedges[i][-1]
            )
            buckets[i] = torch.bucketize(clamped, self.binedges[i]) - 1
        return tuple(buckets)


class BinningAdaptiveSize(BaseBinning):
    def __init__(self, device, n_bins, edges=None, use_bincenters=False, normalize=False, symmetric=False):
        super().__init__(device, n_bins, use_bincenters)
        self.normalize = normalize
        self.symmetric = symmetric

    def reset_edges(self, x):
        l = len(x)
        edges = [None] * l
        for i in range(l):
            if self.symmetric and not self.normalize:
                edges[i] = torch.tensor(
                    [-x[i].abs().max() - 1, x[i].abs().max() + 1], device=self.device
                )
            if self.symmetric and self.normalize:
                edges[i] = torch.tensor([-1, 1], device=self.device)

            else:
                edges[i] = torch.tensor(
                    [x[i].min() - 1, x[i].max() + 1], device=self.device
                )
        self.edges = edges
        self.binedges = [
            torch.linspace(
                e[0], e[1], self.n_bins[i] + 1, dtype=torch.double, device=self.device
            )
            for i, e in enumerate(self.edges)
        ]

    def get_buckets(self, x):
        buckets = [None] * len(x)
        if self.normalize:
            x_normalized = []
            for tensor in x:
                tensor_max = tensor.abs().max(dim=0, keepdim=True)[0]
                normalized_tensor = tensor / tensor_max
                x_normalized.append(normalized_tensor)
            x = x_normalized

        self.reset_edges(x)
        for i in range(len(x)):
            clamped = torch.clamp(
                x[i], self.binedges[i][0] + 1e-04, self.binedges[i][-1]
            )
            buckets[i] = torch.bucketize(clamped, self.binedges[i]) - 1
        return tuple(buckets)


class BinningMaxEntropy(BinningFixedSize): # Pretty sure that this is wrong!!
    def __init__(self, device, n_bins, edges, use_bincenters=False):
        super().__init__(device, n_bins, edges, use_bincenters)

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
        # if normalize:
        #     x_normalized = []
        #     for tensor in x:
        #         tensor_min = tensor.min(dim=0, keepdim=True)[0]
        #         tensor_max = tensor.max(dim=0, keepdim=True)[0]
        #         normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        #         x_normalized.append(normalized_tensor)
        #     x = x_normalized
        self.reset_edges(x, normalize=normalize)
        for i in range(len(x)):
            clamped = torch.clamp(
                x[i], self.binedges[i][0] + 1e-04, self.binedges[i][-1]
            )
            buckets[i] = torch.bucketize(clamped, self.binedges[i]) - 1
        return tuple(buckets)
