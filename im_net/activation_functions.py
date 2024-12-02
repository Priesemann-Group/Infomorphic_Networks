import torch
from torch import nn
from torch.distributions import categorical

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

class ProbabilisticSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        theta = torch.sigmoid(x)#.detach()
        thetas = torch.stack([1-theta, theta], dim=2)
        out = categorical.Categorical(thetas).sample() 
        return out.type(torch.get_default_dtype()) # Sample here as the backward pass also includes the sampling

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
        return sigmoid_grad * grad_output, None, None # also is just the gradient, so I can use the other function

class ActivationWithSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.has_surrogate = True

class BinaryActivationFunction(nn.Module):
    def __init__(self):
        self.has_surrogate = False
        super().__init__()
        


class GraetzActivation(BinaryActivationFunction):
    def __init__(self, output_size, k1=.5, k2=2):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, r, c):
        theta = torch.sigmoid(r*(self.k1+(1-self.k1)*2*torch.sigmoid(self.k2*r*c)))
        theta = theta.clamp(min=1e-6, max=1-1e-6) # to avoid numerical issues
        return torch.stack([1-theta, theta], dim=2)
    
class GraetzActivationHeaviside(ActivationWithSurrogate):
    def __init__(self, output_size, k1=.5, k2=2):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, r, c):
        x = r*(self.k1+(1-self.k1)*2*torch.sigmoid(self.k2*r*c))
        out = HeavisideSigmoidFunction.apply(x)
        return out

class SumActivation(BinaryActivationFunction):
    def __init__(self, output_size, k1=0):
        super().__init__()
    def forward(self, r, c):
        theta = torch.sigmoid(r+0.1*c)
        return torch.stack([1-theta, theta], dim=2)

class KayPhillipsActivation(BinaryActivationFunction):
    def __init__(self, output_size, k1=.5, k2=2):
        print("KayPhillipsActivation has not been tested yet.")
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, r, c):
        theta = torch.sigmoid(r*(self.k1+(1-self.k1)*torch.exp(self.k2*r*c))) #  Note that here is a factor 2 less compared to Graetz
        return torch.stack([1-theta, theta], dim=2)

class LinearBinaryActivation(BinaryActivationFunction):
    def __init__(self, output_size):
        super().__init__()
        print("Should not work, since theta is unbounded here.")

    def forward(self, r, c):
        theta = r+c 
        return torch.stack([1-theta, theta], dim=2)
    
class Schneider3SourceActivation(BinaryActivationFunction):
    def __init__(self, output_size, c1=1./3., c2=1./3., w1=2., w2=2.):
        super().__init__()
        self.c1 = c1 #context impact
        self.c2 = c2 #context impact
        self.w1 = w1
        self.w2 = w2

    def forward(self, r, c, l):
        theta = torch.sigmoid(r*((1 - self.c1 - self.c2) + self.c1 * torch.sigmoid(self.w1 * r * c) + self.c2 * torch.sigmoid(self.w2 * r * l)))
        return torch.stack([1-theta, theta], dim=2)
    
class Schneider3SourceActivationBackprop(ActivationWithSurrogate):
    def __init__(self, output_size, c1=1./3., c2=1./3., w1=2., w2=2.):
        super().__init__()
        self.c1 = c1 #context impact
        self.c2 = c2 #context impact
        self.w1 = w1
        self.w2 = w2

    def forward(self, r, c, l):
        x = r*((1 - self.c1 - self.c2) + self.c1 * torch.sigmoid(self.w1 * r * c) + self.c2 * torch.sigmoid(self.w2 * r * l))
        out = ProbabilisticSigmoidFunction.apply(x)
        return out 

class ThreeSum(BinaryActivationFunction):
    def __init__(self, output_size, c1=1./3., c2=1./3., w1=2., w2=2.):
        super().__init__()

    def forward(self, r, c, l):
        theta = torch.sigmoid(r+0.1*c+l)
        return torch.stack([1-theta, theta], dim=2)

class Schneider3SourceActivationHeaviside(ActivationWithSurrogate):
    def __init__(self, output_size, c1=1./3., c2=1./3., w1=2., w2=2.):
        super().__init__()
        self.c1 = c1 #context impact
        self.c2 = c2 #context impact
        self.w1 = w1
        self.w2 = w2

    def forward(self, r, c, l):
        x = r*((1 - self.c1 - self.c2) + self.c1 * torch.sigmoid(self.w1 * r * c) + self.c2 * torch.sigmoid(self.w2 * r * l))
        out = HeavisideSigmoidFunction.apply(x)
        return out
    
class HeavisideSigmoid(nn.Module):
    def __init__(self):
        super(HeavisideSigmoid, self).__init__()

    def forward(self, x):
        return HeavisideSigmoidFunction.apply(x)


