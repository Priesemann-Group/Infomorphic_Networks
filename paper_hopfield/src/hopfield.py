from functools import partial
import sys
from pathlib import Path
#external modules
import hydra
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from im_net import im_model,prob_estim
import im_net.helper_functions as hf

#implements getting and setting basic properties of the weight matrix 
#set self.weights in your own class
class BaseHopfield(nn.Module):
	def __init__(self) -> None:
		super(BaseHopfield,self).__init__()
		self.weights = None
		self.bias = None

	def forward()->torch.Tensor:
		raise NotImplementedError
	
	def set_weights(self,weights:torch.Tensor,bias:torch.Tensor=None)->None:
		with torch.no_grad():
			self.weights[:,:] = weights[:,:]
			if bias is not None:
				self.bias[:] = bias[:]
		return

	def symmetry(self) -> float:
		norm=torch.linalg.matrix_norm
		weights:torch.Tensor=self.weights.detach()
		sym=(weights+torch.t(weights))/2
		asym=(weights-torch.t(weights))/2
		degree=(norm(sym)-norm(asym))/(norm(sym)+norm(asym))
		return degree.item()
	
	def norm(self,rowwise:bool=False)->float|torch.Tensor:
		if rowwise:
			return torch.linalg.norm(self.weights,dim=1)
		return float(torch.linalg.matrix_norm(self.weights))
	
	def set_symmetric(self) -> None:
		with torch.no_grad():
				triangle=torch.triu(self.weights)
				self.weights[:,:]=triangle[:,:]+torch.t(triangle)[:,:]
		return
	
	def remove_sc(self) -> None:
		with torch.no_grad():
			self.weights.fill_diagonal_(0)
		return

	def freeze_external(self,scale:float) -> None:
		with torch.no_grad():
			self.layer1.sources[0].weight.fill_diagonal_(scale)
		self.layer1.sources[0].requires_grad_(False)

	def scale_internal(self,scale:float) -> None:
		with torch.no_grad():
			self.layer1.sources[1].weight *= scale

	def normalize(self, scale:float) -> None:
		print("Warning! Normalize scales all weights.")
		current_scale=self.norm()
		with torch.no_grad():		
			self.weights[:,:]=(scale/current_scale)*self.weights[:,:]
		return
	
	def fixed_point(self,state:torch.Tensor,max_iterations:int=None,
				diagnosis:bool=False) -> torch.Tensor|tuple[torch.Tensor,int,bool]:
		"""
		Evolves the system until a fixed point is reached or for max_iterations steps. 
		"""
		sequential, binary, ordered = self.module_params.sequential, self.module_params.binary, self.module_params.ordered
		if sequential and binary:
			raise ValueError('Invalid parameter combination. Sequential and binary not supported together.')
		if max_iterations is None:
			max_iterations = self.module_params.max_iterations
		device = self.weights.device
		current_state = state
		old_changes = torch.empty(0,1).to(device)
		for i in range(max_iterations):
			if self.module_params.sequential:
				next_state=self.sequential_forward(current_state,ordered)
			else:
				next_state=self(state=current_state,deterministic=True,binary=binary)
			changes = ((next_state-current_state)!=0).nonzero()
			if len(changes)==0: 
				if diagnosis:
					return next_state,i,True
				return next_state
			#a binary variable should revert to the old state if the same entries are changed twice
			if torch.equal(changes,old_changes): 
				if diagnosis:
					return next_state,i,False
				return next_state
			old_changes=changes
			current_state=next_state
		if diagnosis:
			return current_state,i,False
		return current_state
	
class Hopfield(BaseHopfield):
	def __init__(self,module_params) -> None:
		super(Hopfield,self).__init__()
		self.module_params = module_params
		lp = module_params.layer1
		self.layer1 = nn.Linear(lp.network_size,lp.network_size,lp.bias)
		self.weights = self.layer1.weight
		self.bias = self.layer1.bias

	def forward(self,state,deterministic=False,binary=False) -> torch.Tensor:
		field=self.layer1.forward(state)
		if field.isnan().any():
			raise ValueError("NaN in activations.")
		output=torch.sign(field)
		if binary:
			output = (output+1)/2
		zeros=(output.view(-1)==0.).nonzero()
		if len(zeros)>0:
				output.view(-1)[zeros]=(state.view(-1))[zeros]	
		return output
	
	def activations(self,state):
		fields = self.layer1.forward(state)
		return fields

	#eval only, no grad
	def sequential_forward(self,state,ordered=False):
		with torch.no_grad():
			batch=torch.detach(state).clone()
			weights = self.layer1.weight.detach()
			if (bias := self.layer1.bias) is None:
				bias = torch.zeros(weights.shape[0])
			if ordered:
				order = np.arange(batch.shape[1])
			else:
				order = np.random.permutation(batch.shape[1])
			for index in order: #update one neuron for all patterns
				field = torch.inner(weights[index],batch) + bias[index]
				update=torch.sign(field) 
				if len((update==0.).nonzero())>0:
					 #if the result is numerically 0, old state will be retained
					update[(update==0.).nonzero()]=(batch[:,index])[(update==0.).nonzero()]
				batch[:,index]=update
		return batch
	
	def zero_weights(self):
		with torch.no_grad():
			self.layer1.weight[:,:] = 0
			self.layer1.bias[:] = 0
		
class IMHopfield(BaseHopfield):
	def __init__(self,module_params:dict,binning_method:prob_estim.BaseBinning,
			  patterns:torch.Tensor=None) -> None:
		super(IMHopfield,self).__init__()
		self.module_params:dict = module_params
		lp=module_params.layer1
		#backhook decides which weights are trainable, gives the option to remove external scaling or self-connections
		#external backhook
		if lp.freeze_external:
			external_backhook = 0
		else:
			external_backhook = torch.eye(lp.output_size)
		#internal backhook
		if lp.self_connections:
			internal_backhook = 1
		else: 
			internal_backhook = torch.ones(lp.output_size,lp.input_sizes[1]) - torch.eye(lp.output_size)
		backhook = [external_backhook,internal_backhook]
		if lp.activation_params is None:
			lp.activation_params = {}
		self.layer1=im_model.IM_Layer(lp.input_sizes,lp.output_size,lp.activation,binning_method,connections=backhook,
								 biases=lp.biases, discrete_output_values=lp.discrete_output_values,activation_params=lp.activation_params)
		#set variables for BaseHopfield interface
		self.weights = self.layer1.sources[1].weight
		self.bias = self.layer1.sources[1].bias
		self.init_weights(module_params.init,patterns)
		if lp.freeze_external:
			self.freeze_external(module_params.init.external_scale)
		return

	def init_weights(self, init_params:dict,patterns:torch.Tensor=None):
		if not init_params.name=='default':
			initializer = partial(hf.load_module(init_params.name),a=init_params.params.a,b=init_params.params.b)
			#initializer(self.layer1.sources[0].weight)
			#initializer(self.layer1.sources[1].weight)
		if init_params.start_symmetric:
			self.set_symmetric()
		if init_params.hebbian_init:
			with torch.no_grad():
				new_weights = hebb(patterns)
				self.set_weights(new_weights)
		if 'sc_scale' in init_params: #param might not exist
			with torch.no_grad():
				self.layer1.sources[1].weight.fill_diagonal_(init_params.sc_scale)
		if 'internal_scale' in init_params:
			with torch.no_grad():
				self.scale_internal(init_params.internal_scale)
		if 'external_scale' in init_params:
			with torch.no_grad():
				self.layer1.sources[0].weight.fill_diagonal_(init_params.external_scale)
		
		return
	
	def loss(self):
		return self.layer1.loss(self.module_params.layer1.gamma)

	def forward(self, state:torch.Tensor|None, training_signal:torch.Tensor=None,
			  sample:bool=True, deterministic:bool=False, save_for_loss:bool=False,binary=False) -> torch.Tensor:
		"""
		params:
			state: Internal state of the network.
			training_signal: External state of the network.
			sample: Return random outputs, i.e. samples from the probability. If False, return the firing probability.
			deterministic: Return the sign of the activation function instead of sampling.
			save_for_loss: will save the statistics necessary for calculating pid.
		"""
		if binary:
			raise ValueError('Binary not supported by IMHopfield. Use discrete_output_values instead.')
		if state==None:
			state=torch.zeros_like(training_signal)
		if training_signal==None:
			training_signal=torch.zeros_like(state)
		if state.dim()==1: #need 2d tensor to forward
			training_signal=training_signal.unsqueeze(0)
			state=state.unsqueeze(0)
		output = self.layer1.forward([training_signal,state],
							    sample=sample, use_max=deterministic,save_for_loss=save_for_loss)
		return output
	
	#no gradient
	def sequential_forward(self,state:torch.Tensor,ordered=False) -> torch.Tensor:
		with torch.no_grad():
			batch=torch.detach(state).clone()
			weights:torch.Tensor = self.layer1.sources[1].weight.detach()
			if (bias := self.layer1.sources[1].bias) is None:
				bias = torch.zeros(weights.shape[0])
			if ordered:
				order = np.arange(batch.shape[1])
			else:
				order = np.random.permutation(batch.shape[1])
			for index in order:
				field = torch.matmul(weights[index],batch.t()) + bias[index]
				update = torch.sign(field) 
				if len((update==0.).nonzero())>0:
					 #if the result is numerically 0, old state will be retained
					update[(update==0.).nonzero()]=(batch[:,index])[(update==0.).nonzero()]
				batch[:,index]=update
		return batch

	def inversion(self) -> torch.Tensor:
		weights=self.layer1.sources[0].weight
		inversion= torch.sign(torch.diagonal(weights))
		return inversion
	
	def pid_terms(self) -> np.ndarray:
		_, info = self.layer1.loss(return_information=True)
		return info
	
	def gradient_diagnostic(self):
		grad_a, grad_b = self.layer1.loss_diagnostic()
		return grad_a, grad_b


#symmetry and norm are not implemented
class InfomorphicDAM(BaseHopfield):
	def __init__(self,module_params:dict,binning_method:prob_estim.BaseBinning) -> None:
		super(InfomorphicDAM,self).__init__()
		self.module_params=module_params
		lp=module_params.layer1
		#external backhook
		if lp.freeze_external:
			external_backhook = 0
		else:
			external_backhook = torch.eye(lp.output_size)
		#internal backhook
		self.n_inputs = lp.input_sizes[1]**(lp.exponent-1)
		if lp.self_connections:
			internal_backhook = 1
		else: #definition of sc ambiguous, currently only supports exponent = 3
			len = lp.input_sizes[1]
			i = torch.arange(len).view(len, 1, 1)
			j = torch.arange(len).view(1, len, 1)
			k = torch.arange(len).view(1, 1, len)
			W = (i-j)*(i-k)!=0 #False where a self_connection would be created
			internal_backhook = torch.flatten(W,start_dim=1)
		backhook = [external_backhook,internal_backhook] #the parts of the connectivity that are learnable
		self.layer1=im_model.IM_Layer([lp.input_sizes[0],self.n_inputs],lp.output_size,
								lp.activation,binning_method,connections=backhook,
								biases=lp.biases, discrete_output_values=lp.discrete_output_values,
								activation_params={'beta':lp.beta})
		self.init_weights(module_params.initializer)
		self.weights = self.layer1.sources[1].weight

	def init_weights(self, init_params):
		if not init_params.name=='default':
			initializer = partial(hf.load_module(init_params.name),a=init_params.params.a,b=init_params.params.b)
#			initializer(self.layer1.sources[0].weight)
#			initializer(self.layer1.sources[1].weight)
		if 'sc_scale' in init_params:
			with torch.no_grad():
				self.layer1.sources[1].weight.fill_diagonal_(init_params.sc_scale)
		if e_scale:= init_params.external_scale:
			with torch.no_grad():
				self.layer1.sources[0].weight.fill_diagonal_(e_scale)
		if init_params.start_symmetric:
			self.set_symmetric()

	def loss(self):
		return self.layer1.loss(self.module_params.layer1.gamma)
	
	def forward(self, state:torch.Tensor, training_signal:torch.Tensor=None,
			  sample=True, deterministic=False) -> torch.Tensor:
		"""
		state: Internal state of the network.
		training_signal: External state of the network.
		sample: Return random outputs, i.e. samples from the probability. If False, return the firing probability.
		deterministic: Return the sign of the activation function instead of sampling.
		"""
		if state==None:
			state=torch.zeros_like(training_signal)
		if training_signal==None:
			training_signal=torch.zeros_like(state)
		matrix_activations = torch.bmm(state.unsqueeze(2), state.unsqueeze(1))
		flattened_state = torch.flatten(matrix_activations,start_dim=1)
		output = self.layer1.forward([training_signal,flattened_state],sample=sample,use_max=deterministic)
		return output

######################learning rules#######################################
def hebb(patterns:torch.Tensor):
	with torch.no_grad():
		neurons = patterns.size(1)
		weights = torch.zeros((neurons,neurons))
		for pattern in patterns:
			single_hebb = torch.outer(pattern,pattern)
			weights += single_hebb
		weights.fill_diagonal_(0)
	return weights

def trivial(patterns: torch.Tensor):
	"""
	Returns the trivial unity matrix as weights.
	"""
	with torch.no_grad():
		neurons = patterns.size(1)
		weights = torch.zeros((neurons,neurons))
		weights.fill_diagonal_(1)
	return weights

def gardner(patterns:torch.Tensor,weights:torch.Tensor):
	with torch.no_grad():
		weights = weights.detach().clone()
		for pattern in patterns:
			single_hebb = torch.outer(pattern,pattern)
			weights += single_hebb
		weights.fill_diagonal_(0)
	return NotImplementedError


############################simulate Hydra########################################################
def generate_IMHopfield_config(neurons:int=100,goal:list[float]=[1,0,1,0,0],beta=1,
							    external_scale=1,bias=False,sc_scale=None,sequential=False) -> OmegaConf:
	"""
	Creates the OmegaConf necessary for creating an Infomorphic Hopfield model without using Hydra.
	"""
	mp = {
		'max_iterations':neurons, 'sequential':False,
		'initializer':{
			'name': 'default', 'external_scale': external_scale, 'start_symmetric': False
		},
		'layer1':{
			'biases': [False, bias],
			'input_sizes': [neurons,neurons],
			'output_size': neurons,
			'gamma': goal,
			'activation': 'im_net.activation_functions.SumActivation',
			'beta': beta,
			'discrete_output_values': [-1, 1],
			'self_connections': False,
			'freeze_external': False
		}
	}
	if sc_scale is not None:
		mp['initializer']['sc_scale'] = sc_scale
	conf = OmegaConf.create(mp)
	return conf

def generate_basic_config(neurons:int=100,bias=False,sequential=False) -> OmegaConf:
	"""
	Creates the OmegaConf necessary for creating a basic Hopfield model without using Hydra.
	"""
	mp = {
		'layer1':{
			'bias': bias,
			'sequential':sequential,
			'network_size': neurons,
		}
	}
	conf = OmegaConf.create(mp)
	return conf

def load_binning(device='cpu', n_bins=60, edges=20):
	binning_method = prob_estim.BinningAdaptiveSize(device, n_bins=[n_bins,n_bins],
												 edges=[[-edges,edges],[-edges,edges]])
	return binning_method

@hydra.main(config_path="../conf", config_name="basic_config", version_base=None)
def main(cfg):
	neurons=100
	initial=torch.ones((neurons,neurons))
	device=hf.get_device(cfg.params.pref_gpu)
	binning_cls = hf.load_module(cfg.binning_params.name)
	binning_method = binning_cls(device, **cfg.binning_params.params)
	model = IMHopfield(cfg.layer_params,binning_method).to(device)
	input=torch.ones(neurons).to('cpu')
	output=model.forward(input)
	print(output.device)

if __name__ == "__main__":
	main()