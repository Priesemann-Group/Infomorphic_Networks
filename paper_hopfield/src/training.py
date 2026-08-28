import sys
import os
import logging
from pathlib import Path
from time import perf_counter
import math
#external packages
import numpy as np
import fastprogress
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
#packages from Infomorphic
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from src import hopfield
from src.hopfield import IMHopfield, InfomorphicDAM, BaseHopfield, Hopfield
from im_net import datamanager, im_model, plotting, datasets
import im_net.helper_functions as hf

log = logging.getLogger(__name__)
####################set-up######################
def init_run(cfg):
	if 'n_jobs' in HydraConfig.get().sweeper:
		torch.set_num_threads(HydraConfig.get().sweeper.n_jobs)
		log.info(f"Torch using {torch.get_num_threads()} threads.")
	if cfg.params.seed is not None:
		np.random.seed(cfg.params.seed)
		torch.manual_seed(cfg.params.seed)
	if str(HydraConfig.get().mode) == "RunMode.RUN":
		exp_directory = HydraConfig.get().run.dir
	else:
		exp_directory = os.path.join(HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir)
	if not cfg.storage.progress_bars:
		fastprogress.fastprogress.NO_BAR = True
	if cfg.storage.require_comment:
		if str(HydraConfig.get().mode) == "RunMode.RUN":
			comment = input('Comment to run: ')
		elif HydraConfig.get().job.num == 0:
			comment = input('Comment to multirun: ')
		else:
			comment = 'See job number 0'
		#could eventually be expanded to new DM2 class; save models etc.
	else:
		comment = 'no comment'
	dm = datamanager.DataManager(exp_directory=exp_directory,comment=comment,storage_config=cfg.storage) 
	#dm.save_file(__file__)
	return dm
#create dataset plus test- and trainloaders
def init_dataset(m_patterns, n_neurons, num_workers, dataset_params, device='cpu'):
	if dataset_params.correlation == 'none':
		dataset = datasets.HopfieldDataset(m_patterns, n_neurons,dataset_params.p_bernoulli, device)
	elif dataset_params.correlation == 'spatial':
		dataset = datasets.MarkovHopfieldDataset(m_patterns, n_neurons, True, dataset_params.p_bernoulli, device)
	elif dataset_params.correlation == 'temporal':
		dataset = datasets.MarkovHopfieldDataset(m_patterns, n_neurons, False, dataset_params.p_bernoulli, device)
	else:
		raise ValueError(f"Unknown correlation type: {dataset_params.correlation}")
	batch_size = int(np.maximum(1, int(dataset_params.batch_fraction * m_patterns)))
	trainloader = torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=num_workers)
	testloader = torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=num_workers)
	return dataset, trainloader, testloader

def init_model(device, binning_params, model_params, dataset: torch.utils.data.Dataset = None):
	binning_cls = hf.load_module(binning_params.name)
	binning_method = binning_cls(device, **binning_params.params)
	if model_params.name == 'IMHopfield':
		patterns = dataset[:].cpu()
		model = IMHopfield(model_params, binning_method, patterns).to(device)
	elif model_params.name == 'InfomorphicDAM':
		model = InfomorphicDAM(model_params, binning_method).to(device)
	else:
		model = IMHopfield(model_params, binning_method).to(device)
	return model
	

##################complex testing##################

def alternative_pid(model:IMHopfield,device:torch.device,params:DictConfig)-> torch.Tensor:
	standard_binning = model.layer1.binning
	standard_activation = model.layer1.activation
	if 'binning' in params:
		binning_params = params.binning
		binning_cls = hf.load_module(binning_params.name)
		new_binning = binning_cls(device, **binning_params.params)
		model.layer1.binning = new_binning
	if 'activation_function' in params:
		activation_params = params.activation_function
		model.layer1.activation = hf.load_module(activation_params.activation)(
				output_size=model.module_params.layer1.output_size, **activation_params.activation_params
			)
	with torch.no_grad():
			pid = model.pid_terms()
	model.layer1.activation = standard_activation
	model.layer1.binning = standard_binning
	return pid

def pid_alternative_binning(device, model:IMHopfield,binning_params:DictConfig)-> torch.Tensor:
	
	with torch.no_grad():
			pid = model.pid_terms()
	
	return pid

###########storage#######################
def init_storage(epochs:int,dm:datamanager.DataManager,model:IMHopfield|BaseHopfield, device:torch.device,
				 testloader:torch.utils.data.DataLoader,dataset:torch.utils.data.Dataset,
				 cfg:DictConfig) -> None:
	#capacity run
	storage:DictConfig = cfg.storage
	if "capacity" in cfg:
		if storage.capacity:
			n_steps=math.ceil((cfg.capacity.stop - cfg.capacity.start) / cfg.capacity.step)
			dm.allocate_hdf(dset_names=['patterns', 'acc','acc_stochastic'],
				 dset_length=n_steps, group='capacity')	
		return
	#single run
	datapoints = epochs + 1 
	if storage.weights:
		dm.allocate_weight_hdf(model,dset_length=datapoints)
		dm.write_group_dataset(model, 0)
	# work in progress
	if storage.grads:
		#shape = (datapoints, 2, cfg.params.patterns)
		shape = datapoints
		dm.allocate_hdf(dset_names=['x','thetas'],dset_length=shape, group='grads')
		data = dataset[:].cpu()
		output_probabilities = model(data,data,sample=False,save_for_loss=True)
		grad_a,grad_b = model.gradient_diagnostic()
		dm.write_to_dataset(['x','thetas'],[grad_a,grad_b], 0, group='grads')
	if storage.pid:
		shape = (datapoints, 5, cfg.model.layer1.output_size)
		dm.allocate_hdf(dset_names=['layer1'],dset_length=shape, group='pid_atoms')
		with torch.no_grad():
			data = dataset[:].cpu()
			output_probabilities = model(data,data,sample=False,save_for_loss=True)
			pid = model.pid_terms()
		dm.write_to_dataset(['layer1'],[pid], 0, group='pid_atoms')
	if storage.performance:
		dm.allocate_hdf(dset_names=['acc','acc_stochastic','loss'],
				dset_length=datapoints, group='performance')
		acc=test(testloader,model,threshold_test=storage.testing.threshold_test,threshold=storage.testing.threshold)
		acc_stochastic=test_stochastic(testloader,model,storage.testing.fixed_test.iterations,storage.testing.fixed_test.deterministic)
		with torch.no_grad():
			data = dataset[:].cpu()
			model(data,data,save_for_loss=True)
			loss = model.loss().item()
		dm.write_to_dataset(['acc','acc_stochastic','loss'],
					  [acc,acc_stochastic,loss],0,group='performance')
	if storage.properties:
		dm.allocate_hdf(dset_names=['symmetry','norm'],dset_length=datapoints, group='properties')
		symmetry=model.symmetry()
		norm=model.norm()
		dm.write_to_dataset(['symmetry','norm'],[symmetry,norm],0,group='properties')
	if storage.patterns: #save the patterns
		data=dataset[:].cpu()
		h5_dict={'patterns': data.T} 
		dm.save_data(h5_dict)
	if storage.alt_pid:
		if storage.final.pid:
			shape = (1, 4, model.module_params.layer1.output_size)
		else:
			shape = (datapoints, 4, cfg.model.layer1.output_size)
		for m in storage.alt_pids:
			dm.allocate_hdf(dset_names=['layer1'],dset_length=shape, group=m)	
	if 'alternative_pid' in storage.testing:
		if storage.testing.alternative_pid.final_only:
			return
		test_pid = alternative_pid(model,device,storage.testing.alternative_pid)
		shape = (datapoints, 5, model.module_params.layer1.output_size)
		dm.allocate_hdf(dset_names=[storage.testing.alternative_pid.name],dset_length=shape, group='pid_atoms')
		dm.write_to_dataset([storage.testing.alternative_pid.name],[test_pid], 0, group='pid_atoms')
	return

def store_epoch(model:IMHopfield|BaseHopfield,device, optimizer, loss,
				 epoch_id, storage, dm:datamanager.DataManager, testloader=None)->None|tuple:
	if storage.performance and testloader is None:
		raise ValueError("A testloader must be passed to function to save performance during training.")
	if storage.grads:
		grad_a, grad_b = model.gradient_diagnostic()
		dm.write_to_dataset(['x','thetas'],[grad_a,grad_b], epoch_id, group='grads')
	if storage.pid:
		with torch.no_grad():
			pid = model.pid_terms()
		dm.write_to_dataset(['layer1'],[pid], epoch_id, group='pid_atoms')
	if storage.weights:
		dm.write_group_dataset(model, epoch_id)
	if storage.performance:
		acc= test(testloader,model,threshold_test=storage.testing.threshold_test,threshold=storage.testing.threshold)
		acc_stochastic = test_stochastic(testloader,model,
								   storage.testing.fixed_test.iterations,storage.testing.fixed_test.deterministic)
		dm.write_to_dataset(['acc','acc_stochastic','loss'],
						[acc,acc_stochastic,loss],epoch_id,'performance')
	if storage.properties:
		symmetry=model.symmetry()
		norm=model.norm()
		dm.write_to_dataset(['symmetry','norm'],[symmetry,norm],epoch_id,'properties')
	#reimplement saving the model here
	if storage.alt_pid and not storage.final.pid:
		pids = model.layer1.dit_pids(storage.alt_pids)
		for m in storage.alt_pids:
			dm.write_to_dataset(dset_names=['layer1'],data=[pids[m]], index=epoch_id, group=m)
	dm.edit_run_properties(dict(epochs_finished=epoch_id))
	
	if 'alternative_pid' in storage.testing:
		if storage.testing.alternative_pid.final_only: #this might seem awkward, but alternative_pid might not exist
			return
		test_pid = alternative_pid(model,device,storage.testing.alternative_pid)
		dm.write_to_dataset([storage.testing.alternative_pid.name],[test_pid], epoch_id, group='pid_atoms')
	if storage.performance:
		return acc, acc_stochastic
	return

def final_storage(testloader:torch.utils.data.dataloader,model:BaseHopfield,device:torch.device,
				  storage:DictConfig,dm:datamanager.DataManager,dataset=None) -> None:
	if storage.weights and storage.final.weights:
		log.info('Already saving weights at every epoch. Skipping final weights.')
	if storage.pid and storage.final.pid:
		log.info('Already saving pid at every epoch. Skipping final pid.')
	dm.allocate_hdf(dset_names=['acc','noise_fraction','noise_acc'],dset_length=1, group='final')
	if storage.final.acc:
		acc = test(testloader,model,threshold_test=storage.testing.threshold_test,threshold=storage.testing.threshold)
		dm.write_to_dataset(['acc'],[acc],0,group='final')
	if (noise:=storage.final.noise_fraction) is not None:
		noisy_performance = test(testloader,model,noise,threshold_test=storage.testing.threshold_test,threshold=storage.testing.threshold)
		log.info(f'p(flip)={noise},acc: {noisy_performance:.2f}.')
		dm.write_to_dataset(['noise_fraction','noise_acc'],[noise,noisy_performance],0,group='final')
	if storage.final.weights and not storage.weights:
		dm.allocate_weight_hdf(model,dset_length=1)
		dm.write_group_dataset(model, 0)
	if storage.final.pid and not storage.pid:
		shape = (1, 5, model.module_params.layer1.output_size)
		dm.allocate_hdf(dset_names=['layer1'],dset_length=shape, group='pid_atoms')
		with torch.no_grad():
			if dataset: #get pid if no epochs have passed
				data = dataset[:].cpu()
				output_probabilities = model(data,data,sample=False,save_for_loss=True)
			pid = model.pid_terms()
		dm.write_to_dataset(['layer1'],[pid], 0, group='pid_atoms')
	
	if storage.final.pid:
		pids = model.layer1.dit_pids(storage.alt_pids)
		for m in storage.alt_pids:
			dm.write_to_dataset(dset_names=['layer1'],data=[pids[m]], index=0, group=m)

	if 'alternative_pid' in storage.testing:
		if not storage.testing.alternative_pid.final_only:
			return
		test_pid = alternative_pid(model,device,storage.testing.alternative_pid)
		shape = (1, 5, model.module_params.layer1.output_size)
		dm.allocate_hdf(dset_names=[storage.testing.alternative_pid.name],dset_length=shape, group='pid_atoms')
		dm.write_to_dataset([storage.testing.alternative_pid.name],[test_pid], 0, group='pid_atoms')
	
######################basic test and train functions#######################
#randomly flips bits in the patterns, util function for test()
def noisy_patterns(patterns:torch.Tensor,noise_fraction:float) -> torch.Tensor:
	if noise_fraction>1 or noise_fraction<0:
		raise ValueError("'noise_fraction' is not a percentage.")
	if noise_fraction==0:
		return patterns
	length=patterns.size(1)
	with torch.no_grad():
		mask=torch.ones_like(patterns) #mask with dimensions of patterns determines which bits get flipped
		line_mask=torch.ones(length)
		line_mask[:int(length*noise_fraction)]=-1 
		#shuffle line_mask for every row
		for i in range(patterns.size(0)):
			idx = torch.randperm(length)
			mask[i]=line_mask[idx]
		noisy_patterns= torch.mul(patterns,mask)
	return noisy_patterns

#sequential is a property of fixed point function right now.
def test(testloader:torch.utils.data.DataLoader, model:BaseHopfield,noise_fraction:float=0.0,threshold_test=False,threshold=0.95):
	"""
	Evolves dynamics until a fixed point/two-loop is found.
	"""
	if threshold_test:
		return test_threshold(testloader,model,noise_fraction,threshold)
	model.eval()
	sim_function=nn.CosineSimilarity()
	epoch_acc=np.zeros(len(testloader))
	size1=0
	#f_counter=0
	for c,batch in enumerate(testloader): 
		if c==0:
			size1=batch.size(0)
		#batch=batch*model.inversion() this would account for negative external weights...
		original_patterns:torch.Tensor = batch.detach().clone()
		patterns:torch.Tensor = noisy_patterns(batch,noise_fraction)
		result=model.fixed_point(patterns)
		#inverted_result=model.inversion()*result ,see three lines above
		similarity = sim_function(result,original_patterns)
		epoch_acc[c]=np.mean(similarity.detach().cpu().numpy()) #only saves average acc
	if c!=0: #last batch could be smaller, rescale values to get correct average
		size2=batch.size(0)
		epoch_acc[-1]=epoch_acc[-1]*size2/size1
		alpha=(c+1)*size1/(size2+c*size1)
		return np.mean(epoch_acc)*alpha
	#return np.mean(epoch_acc),f_counter/len(testloader.dataset)
	return np.mean(epoch_acc)

def test_threshold(testloader:torch.utils.data.DataLoader, model:BaseHopfield,noise_fraction:float=0.0,threshold=0.95):
	model.eval()
	sim_function=nn.CosineSimilarity()
	epoch_acc=np.zeros(len(testloader))
	size1=0
	#f_counter=0
	for c,batch in enumerate(testloader): 
		if c==0:
			size1=batch.size(0)
		original_patterns:torch.Tensor = batch.detach().clone()
		patterns:torch.Tensor = noisy_patterns(batch,noise_fraction)
		result=model.fixed_point(patterns)
		similarity = sim_function(result,original_patterns)
		#determine fraction of successful recall
		successes = similarity>=threshold
		success_fraction = successes.sum()/batch.size(0)
		epoch_acc[c]=success_fraction 
	if c!=0: #last batch could be smaller, rescale values to get correct average
		size2=batch.size(0)
		epoch_acc[-1]=epoch_acc[-1]*size2/size1
		alpha=(c+1)*size1/(size2+c*size1)
		return np.mean(epoch_acc)*alpha
	#return np.mean(epoch_acc),f_counter/len(testloader.dataset)
	return np.mean(epoch_acc)

#updates a fixed number of time steps, takes average similarity during that time
def test_stochastic(testloader:torch.utils.data.DataLoader,model:Hopfield,
					iterations:int,deterministic:bool=False) -> float:
	model.eval()
	sim_function=nn.CosineSimilarity()
	epoch_acc=np.zeros(len(testloader))
	size1=0
	for c,batch in enumerate(testloader):
		if c==0:
			size1=batch.size(0)
		initial_state=model(batch,deterministic=deterministic)
		current_state=initial_state
		batch_acc=np.zeros(iterations)
		for i in range(iterations):
			next_state=torch.squeeze(model(state=current_state,deterministic=deterministic))
			batch_acc[i]=np.mean(sim_function(next_state,initial_state).detach().cpu().numpy())
			current_state=next_state
		epoch_acc[c]=np.mean(batch_acc)
	if c!=0:
		size2=batch.size(0)
		epoch_acc[-1]=epoch_acc[-1]*size2/size1
		alpha=(c+1)*size1/(size2+c*size1)
		return np.mean(epoch_acc)*alpha
	return np.mean(epoch_acc)

#single training step
def train(dataloader,repetitions,noisy_transmission:bool, model:IMHopfield,optimizer,retain_graph=False):
	model.train()
	for batch in dataloader:
		if noisy_transmission:
			state = model(state=None,training_signal=batch,save_for_loss=True)
		else:
			state = batch
		with torch.no_grad():
			for i in range(repetitions):
				state = model(state,training_signal=batch,save_for_loss=False)
		state = model(state,training_signal=batch,save_for_loss=True)
		loss = model.loss()
		# Backpropagation
		optimizer.zero_grad()
		loss.backward(retain_graph=retain_graph)
		optimizer.step()
	return loss.item()

###################learning algorithms##############
def fixed_learning(epochs:int,model:BaseHopfield,device:torch.device,optimizer,trainloader,
				   params:DictConfig,storage_params:DictConfig,dm:datamanager.DataManager=None,testloader=None):
	main_bar = fastprogress.progress_bar(range(1,epochs + 1))
	for epoch_id in main_bar:
		##check conditions on epoch
		if 'switch_epoch' in params:
			if epoch_id==params.switch_epoch:
				model.freeze_external(params.switch_scale)
				log.info(f"Froze external weights at epoch {epoch_id}/{epochs}.")
		loss=train(trainloader,params.reps,params.noisy_transmission,model,optimizer,storage_params.grads)
		if (norm:=params.normalization) is not None:
			model.normalize(norm)
		if params.simple_symmetric:
			#resets the model to symmetric 'by hand'
			model.set_symmetric()
		#save states
		if dm is not None:
			output=store_epoch(model,device, optimizer, loss,
			   epoch_id, storage_params, dm, testloader)
			if output is not None: #if acc is saved, it's displayed to the progress bar
				main_bar.comment= f"loss:{loss:.2f},acc: {100*output[0]:.2f}%,stoch.:{100*output[1]:.2f}%"
			else:
				main_bar.comment = f'Training, current loss={loss:.2f}'
		#log.info(f"Finished training for epoch {epoch_id}/{epochs}.")
	return	

def threshold_learning(max_epochs:int,threshold:float,model:BaseHopfield,device:torch.device,
					   optimizer,trainloader,testloader,params:DictConfig,storage:DictConfig,
					   checkpoint_epochs=None,dm:datamanager.DataManager=None,master_bar=None):
	threshold_bar = fastprogress.progress_bar(range(1,max_epochs + 1),master=master_bar)
	for epoch_id in threshold_bar:
		loss=train(trainloader,params.reps,params.noisy_transmission,model,optimizer)
		if (norm:=params.normalization) is not None:
			model.normalize(norm)
		if params.simple_symmetric:
			#resets the model to symmetric 'by hand'
			model.set_symmetric()
		if dm is not None:
			store_epoch(model,device ,optimizer,loss
			   ,epoch_id,storage, dm,testloader)
		if epoch_id in checkpoint_epochs and threshold<=1:
			acc = test(testloader,model,threshold_test=storage.testing.threshold_test,threshold=storage.testing.threshold)
			threshold_bar.comment = f"loss={loss:.2f},acc: {acc*100:.2f}% of {threshold*100:.2f}%"
			if acc >= threshold:
				return True,acc
		else:
			threshold_bar.comment= f"loss={loss:.2f}"
	final_acc = test(testloader,model,threshold_test=storage.testing.threshold_test,threshold=storage.testing.threshold)
	return False,final_acc

def capacity_learning(epochs:int,threshold:float,epsilon:float,start:int,step:int,stop:int,
					  device:torch.device,cfg:DictConfig,dm:datamanager.DataManager=None):
	if dm is None and cfg.storage.capacity:
		raise ValueError("A datamanager must be passed to function to save performance during training.")
	start2 = math.ceil(start*cfg.params.neurons)
	step = math.ceil(step*cfg.params.neurons)
	stop2 = math.ceil(stop*cfg.params.neurons)
	m_max = 0
	master_bar= fastprogress.master_bar(range(start2,stop2,step))
	checkpoints = np.logspace(0,np.log10(cfg.params.epochs),cfg.capacity.n_checkpoints,dtype=int)
	for i,m in enumerate(master_bar):
		master_bar.comment = f"current patterns: {m}. max. capacity: {m_max} so far"
		trainloader, testloader, model, optimizer = setup_training(device, cfg, m)
		_,acc = threshold_learning(epochs,threshold,model,device,optimizer,trainloader,testloader,cfg.params,cfg.storage,
							 checkpoint_epochs=checkpoints)
		acc_stochastic = test_stochastic(testloader,model,
								   cfg.storage.testing.fixed_test.iterations,cfg.storage.testing.fixed_test.deterministic)
		master_bar.write(f'Finished. patterns:{m} Final accuracy {100*acc:.2f}%.')
		if cfg.storage.capacity:
			dm.write_to_dataset(['patterns','acc','acc_stochastic'],
						[m,acc,acc_stochastic],i,'capacity')
		if acc>=epsilon:
			m_max = m
		elif cfg.capacity.interrupt:
			return m_max
	return m_max

def bisection_search(epochs:int,threshold:float,low:int,high:int,stop_distance:int,
					  device:torch.device,cfg:DictConfig,guess:float=None,dm:datamanager.DataManager=None):
	"""
	threshold: Success condition for training. float between (0,1). Also checks if acc>=threshold to stop training prematurely.
	low: lower end of binary search
	high: upper end of binary search
	stop_distance: minimum distance to stop search.
	guess: (optional). A first value to check within the interval.
	"""
	##check acc against threshold at log. spaced points
	checkpoints = np.logspace(0,np.log10(cfg.params.epochs),cfg.bisect.n_checkpoints,dtype=int)
	## first guess to speed up search
	if guess is not None:
		trainloader, testloader, model, optimizer = setup_training(device, cfg, guess)
		_,acc = threshold_learning(epochs,guess,model,device,optimizer,trainloader,testloader,cfg.params,cfg.storage,
							 checkpoint_epochs=checkpoints)
		## loop to update endpoints
		if acc>=threshold:
			low = guess
		else:
			high = guess
	##loop for searching capacity
	while (low+stop_distance)<high:
		midpoint = (low+high)//2
		trainloader, testloader, model, optimizer = setup_training(device, cfg, midpoint)
		_,acc = threshold_learning(epochs,threshold,model,device,optimizer,trainloader,testloader,cfg.params,cfg.storage,
							 checkpoint_epochs=checkpoints)
		print(f'{midpoint} patterns - {acc:.2f} acc.')
		## loop to update endpoints
		if acc>=threshold:
			low = midpoint
		else:
			high = midpoint
	## returns lower estimate.
	return low

##helper to init new networks during bisection_search and capacity_learning
def setup_training(device, cfg, patterns:int):
    dataset,trainloader,testloader=init_dataset(patterns,cfg.params.neurons,cfg.params.num_workers,cfg.dataset,device)
    model=init_model(device,cfg.binning,cfg.model,dataset)
    optimizer = hf.load_module(cfg.optim_params.name)(model.parameters(), **cfg.optim_params.params)
    return trainloader,testloader,model,optimizer

@hydra.main(config_path="../conf", config_name="basic_config", version_base=None)
def main(cfg:DictConfig)->int:
	#--init config and datamanager--

	dm=init_run(cfg)
	start_time_seconds=perf_counter()
	#prepare torch
	if cfg.params.dtype=='double':
		torch.set_default_dtype(torch.float64)
	device=hf.get_device(cfg.params.pref_gpu)
	###END INIT####

	# checks if one of multiple mods is enabled in the configs.
	# The options for mods are a. bisect, b. capacity or c. hebb_sweep. Each of these prepares and then calls its own function.
	# If none of these apply, trains a single network.
	if 'hebb_sweep' in cfg:
		memory_loads = np.linspace(cfg.hebb_sweep.start,cfg.hebb_sweep.stop,cfg.hebb_sweep.n_points)
		hebbian_pid_sweep(cfg,device,dm,memory_loads)
		log.info('Finished Hebbian PID sweep.')
		return
	if 'bisect' in cfg:
		bp = cfg.bisect
		n = cfg.params.neurons
		interval = [bp.low,bp.high,bp.interval]
		interval = [int(n*i) for i in interval]
		guess = OmegaConf.select(bp,'guess',default=None)
		if guess is not None:
			guess = int(n*guess)
		m_max = bisection_search(cfg.params.epochs,bp.threshold,*interval,device,cfg,guess)
		log.info(f"maximum patterns m:{m_max}")
		dm.edit_run_properties(dict(m_max=m_max))
		finish_run(cfg.storage.final,dm,start_time_seconds)
		return m_max
	if "capacity" in cfg:
		log.info("starting capacity run")
		init_storage(None,dm,None,device,None,None,cfg)
		cap:DictConfig = cfg.capacity
		m_max = capacity_learning(cfg.params.epochs,cap.threshold,cap.epsilon,
							cap.start,cap.step, cap.stop,
							device,cfg,dm)
		log.info(f"maximum patterns m:{m_max}")
		dm.edit_run_properties(dict(m_max=m_max))
		finish_run(cfg.storage.final,dm,start_time_seconds)
		return m_max
	### Normal run, prepare dataset, train, then evaluate
	#--prepare dataset--
	dataset,trainloader,testloader=init_dataset(cfg.params.patterns,cfg.params.neurons,
											 cfg.params.num_workers,cfg.dataset,device)
	#--prepare the model--
	model=init_model(device,cfg.binning,cfg.model,dataset)
	optimizer = hf.load_module(cfg.optim_params.name)(model.parameters(), **cfg.optim_params.params)
	#storage initialisation
	init_storage(cfg.params.epochs,dm, model,device, testloader, dataset, cfg)
	main_start_time=perf_counter()

	log.info(f"Starting training for {cfg.params.epochs} epochs.")
	if (threshold:=cfg.params.threshold) is not None:
		threshold_learning(cfg.params.epochs,threshold,model,device,
					 optimizer,trainloader,testloader,cfg.params,cfg.storage,dm)
	else:
		fixed_learning(cfg.params.epochs,model,device,optimizer,trainloader,cfg.params,cfg.storage,dm,testloader)
	log.info('finished training.')
	final_storage(testloader,model,device,cfg.storage,dm,dataset)
	# log.info('finished testing.')
	finish_run(cfg.storage.final, dm, start_time_seconds)
	performance = test(testloader,model,threshold_test=cfg.storage.testing.threshold_test,threshold=cfg.storage.testing.threshold)
	log.info(f'Final accuracy is {performance:.2f}.')
	return performance

def finish_run(final_storage, dm, start_time_seconds):
	dm.edit_run_properties(dict(finished=True))
	if final_storage.post_process:
		log.info(f'Post processing')
		plotting.postProcess(dm)
	end_time_seconds=perf_counter()
	runtime_seconds=math.ceil(end_time_seconds-start_time_seconds)
	dm.edit_run_properties(dict(time_seconds=runtime_seconds))

def hebbian_pid_sweep(cfg,device,dm,memory_loads:np.ndarray[float])->None:
	"""
	Initializes networks at each specified memory load and stores PID. Intended for use with Hebbian initialization.
	memory_loads: An array of memory capacities to be run.
	"""
	if not cfg.model.init.hebbian_init:
		log.warning('Function hebbian_pid_sweep should only be used while setting Hebbian init.')
	## init storage
	datapoints = len(memory_loads)
	shape = (datapoints, 5, cfg.model.layer1.output_size)
	dm.allocate_hdf(dset_names=['layer1'],dset_length=shape, group='pid_atoms')
	dm.allocate_hdf(dset_names=['alpha'],dset_length=datapoints,group='alpha')
	## iterate through memory loads
	for epoch_id,alpha in enumerate(memory_loads):
		m_patterns = int(alpha*cfg.params.neurons)
		#--prepare the model and dataset--
		dataset,trainloader,testloader = init_dataset(m_patterns,cfg.params.neurons,
											 cfg.params.num_workers,cfg.dataset,device)
		model=init_model(device,cfg.binning,cfg.model,dataset)		
		with torch.no_grad():
			data = dataset[:].cpu()
			output_probabilities = model(data,data,sample=False,save_for_loss=True)
			pid = model.pid_terms()
		dm.write_to_dataset(['layer1'],[pid], epoch_id, group='pid_atoms')
		dm.write_to_dataset(['alpha'],[alpha], epoch_id,group='alpha')
	return

@hydra.main(config_path="../conf", config_name="basic_config", version_base=None)
def developing(cfg):
	y = bisection_search(500,0.95,1,200,5,torch.device('cpu'),cfg,165)
	print(y)

if __name__ == "__main__":
	output = main()
	# developing()
	output