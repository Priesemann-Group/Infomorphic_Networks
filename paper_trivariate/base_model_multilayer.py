import argparse, os
import torch
import fastprogress
from torch import nn
import hydra
import logging
import time
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from im_net import im_model
from im_net import helper_functions as hf

import custom_utils as ut

log = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)


            
class BaseModel(nn.Module):
    def __init__(self, layer_params, binning_params, optimizer_params, device, hl1_connections=[1, 1], hl2_connections=[1, 1]):#, hl3_connections=[1, 1]):
        super(BaseModel, self).__init__()
        self.device = device
        self.layer_params = layer_params
        ol_connections = [1, torch.eye(layer_params.output_layer.output_size)]
        bin_methods = self.init_prob_estim_methods(binning_params, device)

        self.hidden_layer1 = self.create_im_layer(layer_params.hidden_layer1, hl1_connections, bin_methods)
        self.hidden_layer2 = self.create_im_layer(layer_params.hidden_layer2, hl2_connections, bin_methods)
        # self.hidden_layer3 = self.create_im_layer(layer_params.hidden_layer3, hl3_connections, bin_methods)        
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
        layer_params = list(self.layer_params.values())
        optimizers = []
        if not self.global_opt:
            for idx, optim in enumerate(optim_params):
                #if the number of gammas is equal to the number of sources, then we have a gamma for each source and share the same optimizer-parameters
                if len(layer_params[idx].gamma) == len(layers[idx].sources):
                    for source_idx in range(len(layers[idx].sources)):
                        optimizers.append(hf.load_module(optim_params[optim].type)(layers[idx].sources[source_idx].parameters(), **optim_params[optim].params))
                else:
                    optimizers.append(hf.load_module(optim_params[optim].type)(layers[idx].parameters(), **optim_params[optim].params))
                optimizers.append(hf.load_module(optim_params[optim].type)(layers[idx].parameters(), **optim_params[optim].params))
        else:
            optim = next(iter(optim_params))
            optimizers.append(hf.load_module(optim_params[optim].type)(self.parameters(), **optim_params[optim].params))
        return optimizers
    
    def reset_c_weights_output(self):
        with torch.no_grad():
            self.output_layer.sources[1].weight[:] = torch.nn.Parameter(torch.eye(self.output_layer.output_size).to(self.device))
        # TODO: fix the c weights for the output layer
class GlobalBackpropModel(BaseModel):
    """
    Simple backpropagation model with 1 hidden layer and 1 output layer.
    """
    def __init__(self, layer_params, binning_params, optimizer_params, device):
        self.global_opt = True
        hl_connections = [1, 1, 1-torch.eye(layer_params.hidden_layer1.output_size)]
        super(GlobalBackpropModel, self).__init__(layer_params, binning_params, optimizer_params, device, hl_connections=hl_connections)

    def forward(self, x, rep=2):
        # initial hidden layer output
        hl_output = torch.zeros((x[0].shape[0], self.hidden_layer1.output_size), device=self.device)
        for j in range(rep): 
            hl_output = self.hidden_layer1.forward([x[0], x[1], hl_output], sample=True)
        return self.output_layer.forward([hl_output, x[1]], sample=False)

class InfomorphicContextLateralMultilayer(BaseModel):
    """
    Infomorphic supervised model with lateral connections in the hidden layer.

    Model with triviate (3-input) neurons in the hidden layer.
    Note that the hidden layer is recurrent and only the last time step is used for training.

    1. Hidden layer receives 3 inputs: image, 2nd layers output, lateral outputs
        - x[0]: (feedforward) input image (flattened, fully connected)
        - x[1]: (contextual) output of the 2nd hidden layer
        - x[2]: (lateral) hidden layer output of the previous time step (no self-connections)

    2. Hidden layer receives 3 inputs: 1st hidden layer output, label, lateral outputs
        - x[0]: (feedforward) output of the 1st hidden layer
        - x[1]: (contextual) output of the output layer
        - x[2]: (lateral) hidden layer output of the previous time step (no self-connections)

    2. Output layer receives 2 inputs: hidden layer output, label
        - x[0]: (feedforward) output of the hidden layer (fully connected)
        - x[1]: (contextual) label (1-hot encoded, 1-to-1 connections with fixed value of 1)
    """
    def __init__(self, layer_params, binning_params, optimizer_params, device):
        self.global_opt = False
        s = layer_params.hidden_layer1.output_size
        hl1_connections = [1, 1, ut.gen_sparse_connectivity(s, s, layer_params.hidden_layer1.max_connections)]

        s = layer_params.hidden_layer2.output_size
        hl2_connections = [1, 1, ut.gen_sparse_connectivity(s, s, layer_params.hidden_layer2.max_connections)]
        
        # s = layer_params.hidden_layer3.output_size
        # hl3_connections = [1, 1, ut.gen_sparse_connectivity(s, s, layer_params.hidden_layer3.max_connections)]
        super(InfomorphicContextLateralMultilayer, self).__init__(layer_params, binning_params, optimizer_params, device, hl1_connections=hl1_connections, hl2_connections=hl2_connections)#, hl3_connections=hl3_connections)


    def forward(self, x, rep=8):
        hl1_output = torch.zeros((x[0].shape[0], self.hidden_layer1.output_size), device=self.device)
        hl2_output = torch.zeros((x[0].shape[0], self.hidden_layer2.output_size), device=self.device)
        # hl3_output = torch.zeros((x[0].shape[0], self.hidden_layer3.output_size), device=self.device)
        ol_output = torch.zeros((x[0].shape[0], self.output_layer.output_size), device=self.device)

        for j in range(rep):
            hl1_output = self.hidden_layer1.forward([x[0], x[1], hl1_output], sample=True).detach()
            hl2_output = self.hidden_layer2.forward([hl1_output, x[1], hl2_output], sample=True).detach()
            # hl3_output = self.hidden_layer3.forward([hl2_output, ol_output, hl3_output], sample=True).detach()
            ol_output = self.output_layer.forward([hl2_output, x[1]], sample=True).detach()
        ol_output = self.output_layer.forward([hl2_output, x[1]], sample=False).detach()

        return ol_output

    def loss(self, gamma=None, return_information=False):
        return self.hidden_layer1.loss(gamma=self.layer_params.hidden_layer1.gamma,return_information=return_information),\
            self.hidden_layer2.loss(gamma=self.layer_params.hidden_layer2.gamma,return_information=return_information),\
            self.output_layer.loss(gamma=self.layer_params.output_layer.gamma, return_information=return_information)
    #self.hidden_layer3.loss(gamma=self.layer_params.hidden_layer3.gamma,return_information=return_information),\
            

def trainer(dataloader, optim, model, device, master_bar, update=True, return_batchwise=False):
    if update: model.train()
    epoch_loss, epoch_acc = np.zeros(len(dataloader)), np.zeros(len(dataloader))
    atoms = [[] for _ in model.children()]
    conf_matrix = torch.zeros((model.output_layer.output_size, model.output_layer.output_size))

    for i_batch, (x_r, x_c) in enumerate(dataloader): #fastprogress.progress_bar(dataloader, parent=master_bar)):
        model.reset_c_weights_output()
        for opt in optim: opt.zero_grad(set_to_none=True)
        in_size = model.layer_params.hidden_layer1.input_sizes[0]
        label_size = model.output_layer.output_size
        x_r = x_r.view(-1, in_size).to(device)
        x_c_one_hot = torch.nn.functional.one_hot(x_c.long(), num_classes=label_size).type(torch.get_default_dtype()).to(device)
        pred = model([x_r, x_c_one_hot])
        if model.global_opt:
            loss = torch.nn.CrossEntropyLoss()(pred[:,:,1], x_c_one_hot)
            epoch_loss[i_batch] = loss.item()
            loss.backward()
        else:
            tuples = model.loss(return_information=True)
            ep_loss = 0
            for i, (loss, layer_atoms) in enumerate(tuples):
                atoms[i].append(layer_atoms.detach().cpu())
                if loss.dim() == 0:
                    ep_loss += loss.item()
                    loss.backward(retain_graph= i<len(tuples)-1)
                else:
                    for l in loss:
                        ep_loss += l.item()
                        l.backward(retain_graph=True)
            epoch_loss[i_batch] = ep_loss

        if update:
            for opt in optim: opt.step()
        
        # infomorphic neurons can be inverted because information theory is invariant under this transformation
        pred = hf.correct_inverse_neurons(pred[:,:,1], invert_others=True)

        batch_conf_matrix = confusion_matrix(x_c.cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), labels=range(label_size))
        conf_matrix += torch.as_tensor(batch_conf_matrix, dtype=conf_matrix.dtype)
        epoch_acc[i_batch] = hf.acc_continuous(pred, x_c_one_hot)
    conf_matrix = conf_matrix/conf_matrix.sum(dim=1, keepdim=True)

    #ut.plot_conf_matrix(conf_matrix)
    if not update or not return_batchwise:
        return (
            epoch_loss.mean(),
            epoch_acc.mean(),
            [np.mean(layer_atoms, axis=0) for layer_atoms in atoms],
            conf_matrix
        )
    atoms = [np.stack(layer_atoms) for layer_atoms in atoms]
    return epoch_loss, epoch_acc, atoms, conf_matrix


def valid(dataloader, model, device, master_bar, return_batchwise=False):
    model.eval()
    epoch_loss, epoch_acc = np.zeros(len(dataloader)), np.zeros(len(dataloader))

    with torch.no_grad():
        for i_batch, x in enumerate(dataloader): #fastprogress.progress_bar(dataloader, parent=master_bar)):
            x_r, x_c = x
            in_size = model.layer_params.hidden_layer1.input_sizes[0]
            label_size = model.output_layer.output_size
            x_r = x_r.view(-1, in_size).to(device)
            x_c_one_hot = torch.nn.functional.one_hot(x_c.long(), num_classes=label_size).type(torch.get_default_dtype()).to(device)

            pred = model([x_r, torch.zeros_like(x_c_one_hot)])
            if model.global_opt:
                loss = torch.nn.CrossEntropyLoss()(pred[:,:,1], x_c_one_hot)
                epoch_loss[i_batch] = loss.item()
            else:
                losses = model.loss() 
                for i, l in enumerate(losses):
                    if l.dim() == 0:
                        epoch_loss[i_batch] += l.item()
                    else:
                        epoch_loss[i_batch] += sum([l.item() for l in l])
                # epoch_loss[i_batch] = sum([l.item() for l in losses])

            pred = hf.correct_inverse_neurons(pred[:,:,1], invert_others=True)

            epoch_acc[i_batch] = hf.acc_continuous(pred, x_c_one_hot)
            #master_bar.child.comment = f'Network Acc.: {epoch_acc[i_batch]:.3f}'
    if return_batchwise:
        return epoch_loss, epoch_acc
    return np.mean(epoch_loss), np.mean(epoch_acc)

@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg):    
    # log number of cores
    torch.set_num_threads(4)
    log.info(f"{torch.get_num_threads()=}")
    start_time = time.perf_counter()
    dm = ut.init_exp_dir(cfg.storage, cfg)
    device = ut.get_device(cfg.exp_params.pref_gpu)
    model_mapping = {
        'InfomorphicContextLateralMultilayerModel': InfomorphicContextLateralMultilayer
    }
    if cfg.exp_params.seed is not None:
        np.random.seed(cfg.exp_params.seed)
        torch.manual_seed(cfg.exp_params.seed)
    else:
        np.random.seed()
    trainloader, valloader, testloader = ut.init_dataloaders(cfg.dataset, cfg.exp_params) # uses also seed from exp_params
    ModelClass = model_mapping[cfg.exp_params.model_class]

    if cfg.exp_params.model_class == 'GlobalBackpropModel':
        optim_params = cfg.optim_params_backprop
    else:
        optim_params = cfg.optim_params

    model = ModelClass(cfg.layer_params, cfg.binning_params, optim_params, device=device)
    model.to(device)


    dm.init_hdf(model, len(trainloader))
    optim = model.optimizers

    # set up progress bar 
    log.info(f"Starting training for {cfg.exp_params.epochs} epochs")
    master_bar = fastprogress.master_bar(range(cfg.exp_params.epochs + 1))

    for epoch_id in master_bar:
        train_loss, train_acc, atoms, conf_matrix = trainer(trainloader, optim, model, device, master_bar, return_batchwise=cfg.storage.batchwise, update=False if epoch_id == 0 else True)
        val_loss, val_acc = valid(valloader, model, device, master_bar)
        _, test_acc = valid(testloader, model, device, master_bar)

        # write to storage
        performances = [train_loss, train_acc, val_loss, val_acc, test_acc, list(conf_matrix.flatten())]
        dm.write_to_hdf(epoch_id, model, atoms, performances, optim)

        # update progress bar and log
        master_bar.write(f'Epoch {epoch_id}: Train_loss: {train_loss.mean():.2f}, Train_acc: {train_acc.mean():.3f}, Val_acc: {val_acc:.3f}, Test_acc: {test_acc:.3f}')   
        log.info(f'Finished epoch {epoch_id}/{cfg.exp_params.epochs}')

    log.info(f'Finished training')
    gc.collect()
    end_time = time.perf_counter()
    log.info(f'Training took {end_time-start_time:.2f} seconds')
    return val_acc # this is the value that is used for the optimization

if __name__ == "__main__":
    main()

