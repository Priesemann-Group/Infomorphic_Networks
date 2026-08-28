import logging
import sys
from pathlib import Path
#external packages
import fastprogress
import numpy as np
import hydra
import torch
from torch import nn
#custom packages
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from src import hopfield, training
from im_net import datamanager, im_model, plotting, datasets
import im_net.helper_functions as hf

log = logging.getLogger(__name__)

def test_hebbian(dataloader, model:hopfield.Hopfield,device):
	model.eval()
	sim_function=nn.CosineSimilarity(1)
	epoch_acc=[]
	for batch in dataloader:
		result=model.fixed_point(batch)
		similarity=torch.mean(sim_function(batch,result)).item()
		epoch_acc.append(similarity)
	#return np.mean(epoch_acc),f_counter/len(dataloader)
	return np.mean(epoch_acc)	

@hydra.main(config_path="../conf", config_name="hebb", version_base=None)
def main(cfg):
	dm = training.init_run(cfg)
	device=hf.get_device(cfg.params.pref_gpu)
	log.info(f"Starting Hebbian capacity run.")
	pattern_range=range(cfg.params.start,cfg.params.stop,cfg.params.step)
	progress_bar = fastprogress.progress_bar(pattern_range)
	dm.allocate_hdf(dset_names=['patterns','acc'], dset_length=len(pattern_range), group='capacity')
	for i,num_patterns in enumerate(progress_bar):
		model = hopfield.Hopfield(cfg.hebbian_params).to(device)
		dataset = datasets.HopfieldDataset(num_patterns,cfg.hebbian_params.layer1.network_size)
		testloader = torch.utils.data.DataLoader(dataset,num_patterns,True)
		model.set_weights(hopfield.hebb(dataset[:]))
		acc=test_hebbian(testloader,model,device)
		#log.info(f"Finished assessment for {num_patterns} patterns.")
		dm.write_to_dataset(['patterns','acc'],[num_patterns,acc],i,group='capacity')
		progress_bar.comment=f"Avg acc for {num_patterns} patterns: {acc}"
	log.info(f"Finished Hebbian capacity run.")

if __name__ == "__main__":
	main()
	#developing()