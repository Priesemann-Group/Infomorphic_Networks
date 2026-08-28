import argparse, os
import torch
import fastprogress
import matplotlib.pyplot as plt
from torch import nn, distributions
import hydra
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from im_net import im_model, prob_estim, datamanager, plotting
from im_net import helper_functions as hf
from im_net import activation_functions as af

import custom_utils as ut

log = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

            
class BaseModel(nn.Module):
    def __init__(self, layer_params, binning_params, optimizer_params, device):
        super(BaseModel, self).__init__()
        self.device = device

        self.layer_params = layer_params
        ol_connections = [1, torch.eye(layer_params.output_layer.output_size)]
        bin_methods = self.init_prob_estim_methods(binning_params, device)

        self.output_layer = self.create_im_layer(layer_params.output_layer, ol_connections, bin_methods)
        self.optimizers = self.configure_optimizers(optimizer_params)

    @staticmethod
    def create_im_layer(params, connections, bin_methods):
        """Wrap method to create an IM_Layer."""
        return im_model.IM_Layer(
            params.input_sizes,
            params.output_size,
            params.activation.type,
            binning=bin_methods[params.binning],
            connections=connections, 
            biases=params.bias,
            discrete_output_values=params.discrete_output_values, 
            activation_params=params.activation.params
        )

    def init_prob_estim_methods(self, binning_params, device):
        bin_methods = dict()
        for binning in binning_params:
            bin_methods[binning] = hf.load_module(binning_params[binning].type)(device, **binning_params[binning].params)
        return bin_methods
    
    def configure_optimizers(self, optim_params):
        layers = list(self.children())
        optimizers = []
        if not self.global_opt:
            for idx, optim in enumerate(optim_params):
                optimizers.append(hf.load_module(optim_params[optim].type)(layers[idx].parameters(), **optim_params[optim].params))
        else:
            optim = next(iter(optim_params))
            optimizers.append(hf.load_module(optim_params[optim].type)(self.parameters(), **optim_params[optim].params))
        return optimizers


    def reset_c_weights_output(self):
        with torch.no_grad():
            self.output_layer.sources[1].weight[:] = torch.nn.Parameter(torch.eye(10))
        # TODO: fix the c weights for the output layer

class InfomorphicReadoutModel(BaseModel):
    """
    Infomorphic supervised model without a hidden layer and only a readout layer.
    
    Model with bivariate (2-input) neurons in the output layer.
    
    1. Output layer receives 2 inputs: image, label
        - x[0]: (receptive) input image (flattened, fully connected)
        - x[1]: (contextual) label (one-hot encoded, fully connected)
    """

    def __init__(self, layer_params, binning_params, optimizer_params, device):
        self.global_opt = False
        super(InfomorphicReadoutModel, self).__init__(layer_params, binning_params, optimizer_params, device)

    def forward(self, x):
        return self.output_layer.forward(x, sample=False).detach()

    def loss(self, gamma=None, return_atoms=False):          
        return self.output_layer.loss(gamma=self.layer_params.output_layer.gamma, return_atoms=return_atoms)


def trainer(dataloader, optim, model, device, master_bar, update=True, return_batchwise=False):
    if update: model.train()
    epoch_loss, epoch_acc = np.zeros(len(dataloader)), np.zeros(len(dataloader))
    atoms = [[] for _ in model.children()]
    conf_matrix = torch.zeros((model.output_layer.output_size, model.output_layer.output_size))

    for i_batch, (x_r, x_c) in enumerate(dataloader): #fastprogress.progress_bar(dataloader, parent=master_bar)):
        model.reset_c_weights_output()
        for opt in optim: opt.zero_grad()

        if x_r.shape[1] != 3: #MNIST
            x_r = x_r.view(-1, 28*28).to(device)
        else: #CIFAR
            x_r = x_r.view(-1, 32*32*3).to(device)
        x_c_one_hot = torch.nn.functional.one_hot(x_c.long(), num_classes=10).type(torch.get_default_dtype()).to(device) 

        pred = model([x_r, x_c_one_hot])

        if model.global_opt:
            loss = torch.nn.CrossEntropyLoss()(pred[:,:,1], x_c_one_hot)
            epoch_loss[i_batch] = loss.item()
            loss.backward()
        else:
            loss, layer_atoms = model.loss(return_atoms=True)
            ep_loss = 0
            atoms[0].append(layer_atoms.detach().cpu())
            if loss.dim() == 0:
                ep_loss += loss.item()
                loss.backward()
            else:
                for l in loss:
                    ep_loss += l.item()
                    l.backward(retain_graph=True)
            epoch_loss[i_batch] = ep_loss
        if update:
            for opt in optim: opt.step()
        
        # infomorphic neurons can be inverted because information theory is invariant under this transformation
        pred = hf.correct_inverse_neurons(pred[:,:,1], invert_others=True)

        batch_conf_matrix = confusion_matrix(x_c.cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), labels=range(10))
        conf_matrix += torch.as_tensor(batch_conf_matrix, dtype=conf_matrix.dtype)
        epoch_acc[i_batch] = hf.acc_continuous(pred, x_c_one_hot)
        #master_bar.child.comment = f'Network Acc.: {epoch_acc[i_batch]:.3f}'
    conf_matrix = conf_matrix/conf_matrix.sum(dim=1, keepdim=True)

    if not update or not return_batchwise:
        return (
            epoch_loss.mean(),
            epoch_acc.mean(),
            [np.mean(layer_atoms, axis=0) for layer_atoms in atoms],
            conf_matrix,
        )
    atoms = [np.stack(layer_atoms) for layer_atoms in atoms]
    return epoch_loss, epoch_acc, atoms, conf_matrix


def valid(dataloader, model, device, master_bar, return_batchwise=False):
    model.eval()
    epoch_loss, epoch_acc = np.zeros(len(dataloader)), np.zeros(len(dataloader))

    with torch.no_grad():
        for i_batch, x in enumerate(dataloader): #fastprogress.progress_bar(dataloader, parent=master_bar)):
            x_r, x_c = x
            if x_r.shape[1] != 3: #MNIST
                x_r = x_r.view(-1, 28*28).to(device)
            else: #CIFAR
                x_r = x_r.view(-1, 32*32*3).to(device)
            x_c_one_hot = torch.nn.functional.one_hot(x_c.long(), num_classes=10).type(torch.get_default_dtype()).to(device)

            pred = model([x_r, torch.zeros_like(x_c_one_hot)])
            if model.global_opt:
                loss = torch.nn.CrossEntropyLoss()(pred[:,:,1], x_c_one_hot)
                epoch_loss[i_batch] = loss.item()
            else:
                loss0 = model.loss() 
                epoch_loss[i_batch] = (loss0).item()

            pred = hf.correct_inverse_neurons(pred[:,:,1], invert_others=True)

            epoch_acc[i_batch] = hf.acc_continuous(pred, x_c_one_hot)
            #master_bar.child.comment = f'Network Acc.: {epoch_acc[i_batch]:.3f}'
    if return_batchwise:
        return epoch_loss, epoch_acc
    return np.mean(epoch_loss), np.mean(epoch_acc)


@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg):
    dm = ut.init_exp_dir(cfg.storage, cfg)
    device = ut.get_device(cfg.exp_params.pref_gpu)
    model_mapping = {
        'InfomorphicReadoutModel': InfomorphicReadoutModel,
    }
    # seed random number generators and get dataloaders
    if cfg.exp_params.seed is not None:
        np.random.seed(cfg.exp_params.seed)
        torch.manual_seed(cfg.exp_params.seed)
    trainloader, valloader, testloader = ut.init_dataloaders(cfg.dataset, cfg.exp_params) # uses also seed from exp_params
    # create correct model variant
    # ModelClass = globals()[cfg.exp_params.model_class] # this is the same as the line below but does not work with ray
    ModelClass = model_mapping[cfg.exp_params.model_class]
    model = ModelClass(cfg.layer_params, cfg.binning_params, cfg.optim_params_single, device=device)
    model.to(device)

    optim = model.optimizers
    dm.init_hdf(model, len(trainloader))


    # set up progress bar 
    log.info(f"Starting training for {cfg.exp_params.epochs} epochs")
    master_bar = fastprogress.master_bar(range(cfg.exp_params.epochs + 1))

    for epoch_id in master_bar:
        train_loss, train_acc, atoms, conf_matrix = trainer(trainloader, optim, model, device, master_bar, return_batchwise=cfg.storage.batchwise, update=epoch_id!=0)

        val_loss, val_acc = valid(valloader, model, device, master_bar)
        _, test_acc = valid(testloader, model, device, master_bar)

        performances = [train_loss, train_acc, val_loss, val_acc, test_acc, list(conf_matrix.flatten())]
        dm.write_to_hdf(epoch_id, model, atoms, performances, optim)
        dm.edit_run_properties(dict(epochs_finished=epoch_id))
        # update progress bar and log
        master_bar.write(f'Epoch {epoch_id}: Train_loss: {train_loss.mean():.2f}, Train_acc: {train_acc.mean():.3f}, Val_acc: {val_acc:.3f}, Test_acc: {test_acc:.3f}')    
        log.info(f'Finished epoch {epoch_id}/{cfg.exp_params.epochs}')

    log.info(f'Finished training')
    return val_acc # this is the value that is used for the sweep

if __name__ == "__main__":
    main()

