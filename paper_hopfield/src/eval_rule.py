import logging
import os
import sys
#external packages
import fastprogress
from functools import partial
import numpy as np
import hydra
import torch
from torch import nn
#custom packages
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from src import hopfield, training, tolmachev, mpf
from im_net import datamanager, im_model, plotting, datasets
import im_net.helper_functions as hf

log = logging.getLogger(__name__)

def test(dataloader, model:hopfield.Hopfield,device,noise:float=0.):
	model.eval()
	sim_function=nn.CosineSimilarity(1)
	epoch_acc=[]
	for batch in dataloader:
		noisy_batch:torch.Tensor = training.noisy_patterns(batch,noise)
		result = model.fixed_point(noisy_batch)
		similarity = torch.mean(sim_function(batch,result)).item()
		epoch_acc.append(similarity)
	return np.mean(epoch_acc)

def choose_learning_rule(cfg):
	"""
	Rules implemented: Hebb, DescentL2, Gardner, GardnerKM, Minimum Probability Flow
	Interface for learning rule: In the end only takes patterns and a dict **params.
	"""
	rule_name = cfg.learning_rule.name
	match rule_name:
		case 'Hebb':
			learning_rule = hopfield.hebb
		case 'DescentL2':
			learning_rule = tolmachev.descent_l2
			#deal with bias
			if cfg.learning_rule.bias==True:
				bias = np.zeros((cfg.learning_rule.len))
				learning_rule = partial(learning_rule,biases=bias)
			initial_weight = np.zeros((cfg.learning_rule.len,cfg.learning_rule.len))
			learning_rule = partial(learning_rule,weights=initial_weight)
		case 'Gardner':
			learning_rule = tolmachev.Gardner
			#deal with bias
			bias = np.zeros((cfg.learning_rule.len))
			initial_weight = cfg.learning_rule.eps * np.ones((cfg.learning_rule.len,cfg.learning_rule.len))
			learning_rule = partial(learning_rule,weights=initial_weight,biases=bias)
		case 'GardnerKM':
			learning_rule = tolmachev.Gardner_Krauth_Mezard
			#deal with bias
			bias = np.zeros((cfg.learning_rule.len))
			initial_weight = cfg.learning_rule.eps * np.ones((cfg.learning_rule.len,cfg.learning_rule.len))
			learning_rule = partial(learning_rule,weights=initial_weight, biases=bias)
		case 'DescentExpSI':
			learning_rule = tolmachev.descent_exp_barrier_si
			#deal with bias
			bias = np.zeros((cfg.learning_rule.len))
			initial_weight = cfg.learning_rule.eps * np.ones((cfg.learning_rule.len,cfg.learning_rule.len))
			learning_rule = partial(learning_rule,weights=initial_weight, biases=bias)
		case 'mpf':
			learning_rule = mpf.binary_mpf
		case 'trivial':
			learning_rule = hopfield.trivial
	return learning_rule


def learn_patterns(model:hopfield.BaseHopfield,dataset,learning_rule,rule_params):
	patterns = dataset[:]
	if rule_params.bias:
		weights,bias = learning_rule(patterns=patterns,**(rule_params.params or {}))
		model.set_weights(torch.tensor(weights),torch.tensor(bias))
	else:
		weights = learning_rule(patterns=patterns,**(rule_params.params or {}))
		model.set_weights(torch.tensor(weights))
	return model

@hydra.main(config_path="../conf", config_name="eval_rule", version_base=None)
def main(cfg):
	dm = training.init_run(cfg)
	device=hf.get_device(cfg.params.pref_gpu)
	log.info(f"Starting evaluation of learning rule.")

	learning_rule = choose_learning_rule(cfg)
	pattern_range=range(cfg.params.start,cfg.params.stop,cfg.params.step)
	progress_bar = fastprogress.progress_bar(pattern_range)
	dm.allocate_hdf(dset_names=['patterns','acc'], dset_length=len(pattern_range), group='capacity')
	for i,num_patterns in enumerate(progress_bar):
		#generate data
		dataset = datasets.HopfieldDataset(num_patterns,cfg.params.neurons)
		testloader = torch.utils.data.DataLoader(dataset,num_patterns,True)
		#train model
		model = hopfield.Hopfield(cfg.eval_model).to(device)
		learn_patterns(model,dataset,learning_rule,cfg.learning_rule)
		#eval
		acc = training.test(testloader,model,cfg.testing.noise,cfg.storage.testing.threshold_test,cfg.storage.testing.threshold)
		dm.write_to_dataset(['patterns','acc'],[num_patterns,acc],i,group='capacity')
		progress_bar.comment=f"patterns={num_patterns} : acc.={100*acc:.2f}% "
	log.info(f"Finished evaluating learning rule.")

if __name__ == "__main__":
	main()
	#developing()