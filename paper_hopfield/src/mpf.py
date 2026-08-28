from functools import partial
import sys
from pathlib import Path
#external modules
import fastprogress
import hydra
import numpy as np
import torch
from torch import nn,utils
from omegaconf import OmegaConf
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from im_net import im_model,prob_estim, datasets
import im_net.helper_functions as hf
from src import hopfield, training

# Taken from: https://github.com/dendisuhubdy/mpf-hopfield
# Online, local Minimum Probability Flow (MPF) learning rule
# achieves exponential storage from few samples
#
# code supplement for paper:
#    C. Hillar and N. Tran, Robust exponential memory in Hopfield networks, 2015.
#                 arXiv: http://arxiv.org/abs/1411.4625
#    Exp Storage Paper:  http://www.msri.org/people/members/chillar/files/nature_cliquenet.pdf
#    Python code: http://www.msri.org/people/members/chillar/files/local_mpf_rule.txt
#
# See also:
#    C. Hillar, J. Sohl-Dickstein, K. Koepsell, Efficient and optimal binary Little-Hopfield
#               associative memory storage using minimum probability flow, NIPS (DISCML), 2012
#    http://www.msri.org/people/members/chillar/files/mpf_hopfield.pdf
#
# C. Hillar, May, 2015
#
# Note: Tested to work under Python 2
# 
# [LICENSED FOR ACADEMIC, NON-COMMERCIAL USE ONLY]
#

def mpf_opr_update(X):
    """ e^(x) ~ 1 approximation MPF update rule """
    J = np.zeros((X.shape[1], X.shape[1]))
    theta = np.zeros(X.shape[1])
    X = np.atleast_2d(X)
    for x in X:
        d = (1. - 2. * x)
        J -= np.outer(d, x)
        theta += d
    J[np.eye(X.shape[1], dtype=bool)] *= 0
    return J, theta

def mpf_objective_gradient(X, J, return_obj=False, low=-40, high=40) -> np.ndarray|tuple[np.ndarray,np.ndarray]:
    """ J is a square np.array with -2 * theta on diagonal
        X is a M x N np.array of binary vectors 
        NOTE: This is the MPF objective function / gradient
        with 2J (2 theta) replacing J (theta) in the MPF objective:
            Flow = 1/|X| sum_{x in X} sum_{x' bit flip of x} exp(Ex - Ex')
        This gives the same MIN but is easier to manipulte (no 1/2 in the exp)
        Divide parameters by 2 to get ARGMIN of original MPF objective
        (although unnecessary since dynamics is the same)
    """
    X = np.atleast_2d(X)
    M, N = X.shape
    S = 2 * X - 1
    F = -S * np.dot(X, J.T) + .5 * np.diag(J)[None, :]
    Kfull = np.exp(np.clip(F, low, high))  # to avoid exp overflows
    dJ = -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
    dJ = .5 * (dJ + dJ.T)
    if return_obj:
        return Kfull.sum() / M, dJ / M
    else:
        return dJ / M

def mpf_update(X, J, theta) -> tuple[np.ndarray,np.ndarray]:
    """ full MPF local update rule """
    J[np.eye(J.shape[0], dtype=bool)] = -2 * theta
    DJ = mpf_objective_gradient(X, J)
    Dtheta = -.5 * np.diag(DJ)
    DJ[np.eye(J.shape[0], dtype=bool)] *= 0
    J[np.eye(J.shape[0], dtype=bool)] *= 0
    return DJ, Dtheta

def mpf_step(model:hopfield.Hopfield,trainloader,alpha:float)->None:
	"""
	patterns: The patterns in format (-1,1)
	J: The weights.
	theta: biases.
	alpha: learning rate.
	"""
	J = model.layer1.weight.detach().numpy()
	theta = model.layer1.bias.detach().numpy()
	#get updates
	for batch in trainloader:
		with torch.no_grad():
			X = 0.5 *(batch+1)
		DJ, DT = mpf_update(X, J, theta)
		J -= alpha * DJ
		theta -= alpha * DT
	#put updates back into model
	with torch.no_grad():
		model.layer1.weight[:,:] = torch.tensor(J)[:,:]
		model.layer1.bias[:] = torch.tensor(theta)[:]

def binary_to_bipolar(J:np.ndarray,theta:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    weights = 0.5*J
    bias = - theta + 0.5*J.sum(axis=1)
    return weights, bias

def binary_mpf(patterns:np.ndarray,lr:float,epochs,J=None,theta=None)->tuple[np.ndarray,np.ndarray]:
	S = patterns
	X = (S+1)/2
	neurons = patterns.shape[1]
	if J is None: #would properly transform to binary first
		J = np.zeros((neurons,neurons))
	if theta is None:
		theta = np.zeros((neurons))
	progress_bar = fastprogress.progress_bar(range(epochs))
	for j in progress_bar:
		DJ, DT = mpf_update(X, J, theta)
		J -= lr * DJ
		theta -= lr * DT
	J = (J + J.T) / 2
	weights, bias = binary_to_bipolar(J,theta)
	return weights, bias

######################My custom implementation####################################################
def energy(model:hopfield.Hopfield, batch):
	activation = model.activations(batch)
	energy = -torch.linalg.vecdot(batch,activation)
	return energy

def MPF_loss(model,batch:torch.Tensor):
	"""
	batch: (b,N)
	"""
	#generate all single flips
	n_neurons = batch.size()[-1]
	diagonal_flips = 2 * torch.diag_embed(-batch)
	single_flips = batch[:,:,None].expand(-1,-1,n_neurons)
	single_flips = single_flips + diagonal_flips #(b,N,N)

	E_baseline = energy(model,batch) # (b,1)
	E_comparison = energy(model,single_flips) #(b,N)

	#p_flow = torch.exp(-(E_baseline[:,None]-E_comparison)).sum()
	p_flow = (1/n_neurons)*torch.exp(E_baseline[:,None]-E_comparison).sum()
	#unfinished, but should work (at least as pseudocode)
	return p_flow

def logistic_regression_loss(model:hopfield.Hopfield,batch:torch.Tensor):
	p_firing = torch.sigmoid(model.activations(batch))
	p_target = (1/2)*(batch+1)
	loss_fct = torch.nn.BCELoss()
	loss = loss_fct(p_firing,p_target)
	return loss

def optim_step(model,trainloader,optimizer,cfg):
	for batch in trainloader:
		match cfg.learning_rule.name:
			case 'mpf':
				loss = MPF_loss(model,batch)
			case 'regression':
				loss = logistic_regression_loss(model,batch)
			case _:
				raise ValueError('No valid learning rule selected.')
		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return loss

@hydra.main(config_path="../conf", config_name="eval_rule", version_base=None)
def main(cfg):
	num_patterns = 160
	dm = training.init_run(cfg)
	device=hf.get_device(cfg.params.pref_gpu)
	
	dataset = datasets.HopfieldDataset(num_patterns,cfg.params.neurons)
	trainloader = utils.data.DataLoader(dataset,num_patterns,True)
	#train model
	model = hopfield.Hopfield(cfg.eval_model).to(device)
	model.set_symmetric()
	model.zero_weights()
	optimizer = hf.load_module(cfg.optim_params.name)(model.parameters(), **cfg.optim_params.params)
	if cfg.learning_rule.name=='mpf':
			weights, bias = binary_mpf(trainloader.dataset[:],**cfg.learning_rule.params)
			model.set_weights(torch.tensor(weights),torch.tensor(bias))
			acc = training.test(testloader=trainloader,model=model)
			stochastic_acc = training.test_stochastic(trainloader,model,10)
			symmetry = model.symmetry()
			print(f'acc={acc:.2f},acc_2:{stochastic_acc:.2f},sym:{symmetry}')
			return
	optim_bar = fastprogress.progress_bar(range(cfg.learning_rule.epochs))
	for i in optim_bar:
		loss = optim_step(model,trainloader,optimizer,cfg)
		if cfg.learning_rule.sc==False:
			model.remove_sc()
		acc = training.test(testloader=trainloader,model=model)
		stochastic_acc = training.test_stochastic(trainloader,model,10)
		symmetry = model.symmetry()
		optim_bar.comment = f'acc={acc:.2f}, loss={loss:.2f},acc_2:{stochastic_acc:.2f}'
	acc = training.test(testloader=trainloader,model=model)
	print(f"Final acc: {acc:.2f}")



if __name__ == "__main__":
	main()