import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
from copy import deepcopy
from scipy.optimize import minimize

#rules from Tolmachev et Manton (2020), https://arxiv.org/abs/2010.01472
#adapted from https://github.com/ptolmachev/Hopfield_Nets (no license specified upstream)
#see README.md "External Projects" for attribution details, including the fork used for stability plots

def l2norm_difference(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    return (1 / 2) * np.sum((lmbd * h - Z[i, :])**2) + (alpha / 2) * np.sum(weights_and_bias ** 2)


def descent_l2(N, patterns, weights, biases, sc, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} (lmbd h_i^k sigma_i^k - sigma_i^k)^2

    works better with L-BFGS-B
    '''
    jac = egrad(l2norm_difference, 0)
    # hess = jacobian(jac)
    if incremental:
        for i in range(N): #for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100*np.ones(x0.shape[-1]), 100*np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(l2norm_difference, x0, args=(pattern, i, lmbd, alpha),
                               jac = jac,# hess=hess,
                               bounds = bnds,
                               method='L-BFGS-B', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-100*np.ones(x0.shape[-1]),100*np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(l2norm_difference, x0, args=(patterns, i, lmbd, alpha),
                           jac = jac, #hess=hess,
                           bounds=bnds,
                           method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases

def Gardner(N, patterns, weights, biases, sc, lr, k):
    '''
    Gardner rule rule proposed in (1988) The space of interactions in neural network models
    '''
    for i in range(N):  # for each neuron independently
        for j in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ pattern.T + biases[i])
            sum_of_squares = np.sum(weights[i, :] ** 2) + biases[i] ** 2
            y = (h_i * pattern[i])/(np.sqrt(sum_of_squares))
            while (k >= y):
                weights[i, :] = deepcopy(weights[i, :] + lr * pattern[i] * pattern)
                if sc == False:
                    weights[i, i] = 0
                biases[i] = deepcopy(biases[i] + lr * pattern[i])
                h_i = (weights[i, :] @ pattern.T + biases[i])
                sum_of_squares = np.sum(weights[i,:]**2) + biases[i]**2
                y = (h_i * pattern[i])/(np.sqrt(sum_of_squares))
    return weights, biases

def Gardner_Krauth_Mezard(N, patterns, weights, biases, sc, lr, k, maxiter):
    '''
    Gardner rule rule proposed in (1987) Krauth Learning algorithms with optimal stability in neural networks +
    Krauth Mezard update strategy
    '''
    Z = np.array(patterns).T
    M = 0
    p = Z.shape[-1]
    Z_ = np.vstack([Z, np.ones(p)])
    w_and_b = deepcopy(np.hstack([weights, biases.reshape(N, 1)]))
    y_global = ( (w_and_b @ Z_).T/ (np.sqrt(np.sum(w_and_b ** 2, axis=1))) )* Z.T #
    while (np.any(y_global < k) and M < maxiter):
        for i in range(N):  # for each neuron independently
            # compute normalised stability measure (h_i, sigma_i)/|w_i|^2_2
            sum_of_squares = np.sum(weights[i, :] ** 2 + biases[i]**2)
            ys =  ( (weights[i, :] @ Z + biases[i])/ (np.sqrt(sum_of_squares)) )  * Z[i, :] #
            #pick the pattern with the weakest y
            ind_min = np.argmin(ys)
            weakest_pattern = np.array(deepcopy(patterns[ind_min].reshape(1, N)))
            h_i = (weights[i, :].reshape(1, N) @ weakest_pattern.T + biases[i]).squeeze()
            # if the new weakest pattern is not yet stable with the margin k
            y = (h_i * weakest_pattern[0, i])/(np.sqrt(sum_of_squares)) #
            while (y < k):
                weights[i, :] = deepcopy(weights[i, :] + lr * (weakest_pattern[0, i] * weakest_pattern).squeeze())
                #set diagonal elements to zero
                if sc == False:
                    weights[i, i] = 0
                biases[i] = biases[i] + lr * weakest_pattern[0, i]
                sum_of_squares = np.sum(weights[i, :] ** 2 + biases[i] ** 2)
                h_i = (weights[i, :].reshape(1, N) @ weakest_pattern.T + biases[i]).squeeze()
                y = (h_i * weakest_pattern[0, i])/(np.sqrt(sum_of_squares)) #
        w_and_b = deepcopy(np.hstack([weights, biases.reshape(N, 1)]))
        y_global = ( (w_and_b @ Z_).T/ (np.sqrt(np.sum(w_and_b ** 2, axis=1))) )* Z.T #
        M += 1
        if M >= maxiter:
            print('Maximum number of iterations has been exceeded')
    return weights, biases

def sum_exp_barriers_si(weights_and_bias, patterns, i, lmbd):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    return np.sum(np.exp((-lmbd * h * Z[i, :]) / np.sqrt(np.sum(weights_and_bias ** 2)))) + 0.1*(np.sum(weights_and_bias ** 2) - 1)**2

def descent_exp_barrier_si(N, patterns, weights, biases, sc, incremental, tol, lmbd):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} -(h_i^{\mu} \sigma_i^{\mu}) / (\sum_j w_{ij}^2 + b_i^2)^(0.5)

    comment: for some reson L-BFGS-B without a hessian works much faster than Newton-CG with Hessian!
    '''
    jac = egrad(sum_exp_barriers_si, 0)
    # hess = jacobian(jac)
    if incremental:
        for i in range(N):  # for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100*np.ones(x0.shape[-1]), 100*np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(sum_exp_barriers_si, x0, args=(pattern, i, lmbd),
                               jac= jac,# hess = hess,
                               bounds = bnds,
                               method='L-BFGS-B', tol=tol, options={'disp': False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] = deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N):  # for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-np.ones(x0.shape[-1]),np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(sum_exp_barriers_si, x0, args=(patterns, i, lmbd),
                           jac= jac, #hess = hess,
                           bounds = bnds,
                           method='L-BFGS-B', tol=tol, options={'disp': False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] = deepcopy(res['x'][-1])
    return weights, biases