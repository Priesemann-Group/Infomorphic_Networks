import math
import numpy as np
import optuna
import pandas as pd
from scipy import integrate, optimize
import torch

import im_net.datamanager as dm
###########CONSTANTS#############
EXTERNAL=0
INTERNAL=1
REDUNDANCY=2
SYNERGY=3
ENTROPY=4
atom_names = ['Target','Recurrent','Redundancy','Synergy','Entropy']
atom_names_latex = [r'$\Pi_{r}$',r'$\Pi_{l}$',r'$\Pi_{red.}$',r'$\Pi_{syn.}$',r'$H_{res}$']
classical_names = ['ext_mutual','int_mutual','double mutual','redundancy','entropy']

######access stored data conveniently in numpy format
def filter_dm(datamanager:dm.DataManager,filter_by:str=None,condition:int|bool=None):
	filteredFrame=datamanager.sel[datamanager.sel[filter_by]==condition]
	datamanager.sel = filteredFrame
	return

def sort_dm(datamanager:dm.DataManager,sort_by:str):
	sortedFrame=datamanager.sel.sort_values(by=sort_by)
	datamanager.sel = sortedFrame 
	return

def list_of_dicts_to_arrays(ls_dict:list[dict],keys=None,squeeze=False) -> np.ndarray:
	keys = ls_dict[0].keys() if keys is None else keys
	list_data = [[entry[k] for k in keys] for entry in ls_dict]
	array_list = [np.array(entry) for entry in list_data]
	if squeeze:
		array_list = [np.squeeze(array) for array in array_list]
	return array_list

def get_all_attributes(data_manager:dm.DataManager):
	"""Returns all attributes in first(!) run."""
	groups = data_manager.list_selected_datasets()[0]
	attributeDict = {group: data_manager.load_selected(group)[0].keys() for group in groups}
	# attributes = groupList[0].keys()
	return attributeDict

#todo: replace first half with list_of_dicts_to_array
def group_data_to_numpy(data_manager:dm.DataManager,group:str,data_variable:str,tag:str=None) -> np.ndarray:
	#sort and filter
	groupList= data_manager.load_selected(group)
	performanceDict = {key: [dic[key] for dic in groupList] for key in groupList[0]}
	data=pd.DataFrame(performanceDict[data_variable]).fillna(0)
	array=data.to_numpy()
	if tag is not None:
		tags=data_manager.sel[tag]
		return array,tags
	return array

def pid_to_numpy(data_manager,mean=False,variable:str='layer1'):
	"""
	returns: Numpy Array of shape [atom][run][time]~[neuron]~. The last dimension is optional.
	"""
	group = 'pid_atoms'
	groupList = data_manager.load_selected(group)
	performanceDict = {key: [dic[key] for dic in groupList] for key in groupList[0]}
	performanceArray = np.array(performanceDict[variable])
	performanceArray = performanceArray.transpose((2,0,1,3))
	if mean:
		return np.mean(performanceArray,-1)
	return performanceArray

def get_patterns(dm:dm.DataManager) -> np.ndarray:
	return dm.load_selected_dict('/','patterns')['patterns']

##################get weights#############################################

SOURCES = ['sources.0.weight','sources.1.weight']
def reconstruct_weights_epoch(weight_dict):
	"""Assumes layer will have two sources."""
	keys = sorted(weight_dict.keys(),key=int)
	lengths = [weight_dict[keys[0]][source].shape[-1] for source in SOURCES]
	tensor_a = torch.Tensor(len(keys),lengths[0])
	tensor_b = torch.Tensor(len(keys),lengths[1])
	#assign two tensors of appropriate size
	for i,key in enumerate(keys):
		#fill in both tensors in line by line
		tensor_a[i] = torch.Tensor(weight_dict[key][SOURCES[0]][0])
		tensor_b[i] = torch.Tensor(weight_dict[key][SOURCES[1]][0])
	return tensor_a, tensor_b

def get_weights(dm:dm.DataManager):
	all_dicts = dm.load_selected_rec('model_weights')
	list_of_weights = [reconstruct_weights_epoch(single_dict['layer1']) for single_dict in all_dicts]
	weights_a = [both_weights[0] for both_weights in list_of_weights]
	weights_b = [both_weights[1] for both_weights in list_of_weights]
	return weights_a, weights_b

def get_final_weights(dm:dm.DataManager):
	all_dicts = dm.load_selected_rec('model_weights')
	weights_a, weights_b = reconstruct_weights_epoch(all_dicts[-1]['layer1'])
	return weights_a, weights_b

###convert back and forth between PID and classical######
transition_matrix = [[1,0,1,0,0],#ext_mutual
					 [0,1,1,0,0],#int_mutual
					 [1,1,1,1,0],#double_mutual
					 [0,0,1,0,0],#redundancy
					 [0,0,0,0,1]]#h_res
inverse_transition = np.linalg.inv(transition_matrix)
def pid_to_classical(input:np.array,axis=0) -> np.array:
	assert input.shape[axis]==5
	if input.ndim!=2:
		raise ValueError('Input needs to be 2-dimensional.')
	if axis==1:
		input = input.T
	output = np.matmul(transition_matrix,input)
	if axis==1:
		output=output.T
	return output

def classical_to_pid(input:np.array,axis=0) -> np.array:
	assert input.shape[axis]==5
	if input.ndim!=2:
		raise ValueError('Input needs to be 2-dimensional.')
	if axis==1:
		input = input.T
	output = np.matmul(inverse_transition,input)
	if axis==1:
		output=output.T
	return output

#################common patterns##########################
def quick_load(file_name:str,group_x:str|None,var_x:str,group_y:str,var_y:str,parameter:str=None)->tuple[dm.DataManager,np.ndarray,np.ndarray]:
	"""
	Loads up to two quantities of the run just from the file name.
	group_x: if None, var_x is assumed to be a parameter of the run.
	"""
	data_manager = dm.DataManager(file_name,
										mode='analysis', add_run_properties=True, verbose=0)
	if parameter is not None:
		sort_dm(data_manager,parameter)
	if group_x is None:
		x = data_manager.sel[var_x]
	else:
		x = group_data_to_numpy(data_manager,group_x,var_x)
	y = group_data_to_numpy(data_manager,group_y,var_y)
	
	if parameter is not None:
		label = data_manager.sel[parameter]
		return data_manager,x,y,label
	return data_manager,x,y

def load_capacity_runs(file_name:str,parameter:str=None)-> tuple[dm.DataManager,np.ndarray,np.ndarray]|tuple[dm.DataManager,np.ndarray,np.ndarray,np.ndarray]:
	"""
	Limitation, gives error if multiple n_neurons are used.
	parameter: Sort outputs by value. String.
	Returns: datareader, memory load (x), capacity(y),(optional:labels)
	"""
	params = {
		'group_x' :'capacity',
		'group_y' : 'capacity',
		'var_x' : 'patterns',
		'var_y' : 'acc'
	}
	output = quick_load(file_name=file_name,parameter=parameter,**params)
	alpha = output[1]/output[0].sel['params.neurons'][0]
	if parameter is None:
		return output[0],alpha,output[2]
	return output[0],alpha,output[2],output[3]

###################################handle databasis#############################################
def pre_process_location(location:str):
	if len(location)>= 10 and location[:10]=='sqlite:///':
		return location
	processed_string = f'sqlite:///{location}'
	return processed_string

def get_studies(storage):
	studies = optuna.study.get_all_study_summaries(storage=storage)
	study_names = [study.study_name for study in studies]
	return study_names

def load_study(storage_file,study_name=None):
	"""
	Loads study from database into a dataframe. Defaults to first study if no name is given.
	"""
	storage = pre_process_location(storage_file)
	if not study_name:
		try:
			study_name = get_studies(storage)[0]
		except:
			raise ValueError(f'Study could not be loaded.')
	study = optuna.study.load_study(study_name=study_name,storage=storage)
	return study

def load_df_from_db(storage_file,study_name = None):
	"""
	Loads study from database into a dataframe. Defaults to first study if no name is given.
	"""
	study = load_study(storage_file,study_name)
	df = study.trials_dataframe()
	return df

###########perform calculations for analysis#############

def gamma_values(weights:np.ndarray,patterns:np.ndarray)-> np.ndarray:
	local_fields=(weights @ patterns)
	aligned_fields=np.multiply(local_fields,patterns)
	rowwise_norm=np.linalg.norm(weights,axis=1)
	gamma=aligned_fields/rowwise_norm[:,None]
	return gamma

#refer to The space of interactions in neural network models, E. Gardner 1988
def capacity_gardner (threshold):
	if threshold < 0:
		raise ValueError
	func= lambda t: math.exp(-t**2/2)*(t+threshold)**2/math.sqrt(2*math.pi)
	output,error=integrate.quad(func,-threshold,np.inf)
	if error>1e-2:
		print('Integration could have gone wrong.')
	capacity=1/output
	return capacity

def threshold_gardner(capacity):
	func = lambda k: capacity_gardner(k)-capacity
	threshold = optimize.root_scalar(func,bracket=[0,10]).root
	return threshold

#####################deprecated#########################
#depecrated, for use with old DataManager
def pid_to_numpy_old(data_manager:dm.DataManager,atoms:int|list[int]=[0,1,2,3,4],average:bool=True) -> np.ndarray:
	"""
	returns: Numpy Array of shape [atom][run][time]([neuron])
	"""
	informationList= data_manager.load_selected_rec('info_quantities')
	informationList2 = [dic['layer1'] for dic in informationList] 
	informationArray=np.array([list(dic.values()) for dic in informationList2])#format is [run][neuron][time][pid]
	orderedArray=np.transpose(informationArray,[3,0,2,1]) #format is [atom][run][time][neuron]
	if average:
		processedArray=np.mean(orderedArray,-1)
	else:
		processedArray=orderedArray
	return processedArray[atoms]