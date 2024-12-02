import h5py
import os
import numpy as np
import torch
import yaml
import pandas as pd
import copy
import im_net.prob_estim as prob_estim
import im_net.helper_functions as hf
import subprocess
from hydra import compose, initialize, initialize_config_module, initialize_config_dir
from omegaconf import OmegaConf

from hydra.core.global_hydra import GlobalHydra


# class DataManager:
#     def __init__(
#         self,
#         exp_directory,
#         mode="training",
#         save_model=True,
#         cp_spacing="log",
#         cp_number=None,
#         cp_dir="checkpoints",
#         epochs=None,
#         add_run_properties=False,
#         verbose=0,
#     ):
#         self.exp_directory = exp_directory
#         self.verbose = verbose
#         self.cp_dir = cp_dir
#         if mode == "training":
#             # in the future, this should not be necessary, but for now, we need to create the spacing here (better is to get it from a scheduler)
#             assert epochs is not None
#             assert cp_number is not None
#             self.epochs = epochs
#             if cp_spacing == "log":
#                 self.checkpoints = np.unique(
#                     np.logspace(0, np.log10(epochs + 1), cp_number, dtype=int) - 1
#                 )  # log spaced checkpoints. Could be less than checkpoints
#             elif cp_spacing == "linear":
#                 self.checkpoints = np.unique(
#                     np.linspace(0, epochs + 1, cp_number, dtype=int, endpoint=True)
#                 )
#             elif cp_spacing == "all":
#                 self.checkpoints = np.arange(epochs + 1)
#             if save_model:
#                 os.makedirs(os.path.join(self.exp_directory, cp_dir), exist_ok=True)
#             user_email = (
#                 subprocess.run(["git", "config", "user.email"], stdout=subprocess.PIPE)
#                 .stdout.strip()
#                 .decode()
#             )
#             commit_hash = (
#                 subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
#                 .stdout.strip()
#                 .decode()
#             )
#             properties = dict(
#                 user_email=user_email,
#                 commit_hash=commit_hash,
#                 finished=False,
#                 comment="",
#                 epochs_finished=0,
#             )
#             with open(
#                 os.path.join(self.exp_directory, "run_properties.yaml"), "w"
#             ) as f:
#                 yaml.dump(properties, f)

#         elif mode == "load":
#             pass

#         elif mode == "analysis":
#             self.all = self.load_configs(
#                 add_run_properties=add_run_properties, verbose=verbose
#             )
#             self.sel = copy.copy(self.all)  # select all by default

#     def edit_run_properties(self, changes=None):
#         with open(os.path.join(self.exp_directory, "run_properties.yaml"), "r") as f:
#             properties = yaml.load(f, Loader=yaml.FullLoader)
#         for key, value in changes.items():
#             properties[key] = value
#         with open(os.path.join(self.exp_directory, "run_properties.yaml"), "w") as f:
#             yaml.dump(properties, f)

#     def rerun_failed_runs(self, num_cpus=2, ):
#         failed_runs = self.all[self.all.finished == False]
#         raise NotImplementedError 
    

#     def alloc_group_hdf(
#         self, model, dset_length, group="model_weights", file_name="data.h5"
#     ):
#         assert group in ["model_weights", "info_quantities"]
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             f.require_group(group)
#             for layer_name, layer in model.named_children():  # layer
#                 f[group].require_group(layer_name)
#                 if group == "model_weights":
#                     for param_name, param in layer.named_parameters():
#                         for j in range(param.data.cpu().numpy().shape[0]):
#                             f[group][layer_name].require_group(str(j))
#                             f[group][layer_name][str(j)].create_dataset(
#                                 param_name,
#                                 (dset_length, param.data.cpu().numpy()[j].size),
#                             )
#                 elif group == "info_quantities":
#                     for j in range(layer.output_size):
#                         f[group][layer_name].create_dataset(
#                             str(j), (dset_length, len(layer.info_quantities[j]))
#                         )

#     def alloc_hdf(self, dset_names, dset_length, group="/", file_name="data.h5"):
#         """
#         Creates new dataset in hdf5 file. If the datasets are not of the same length, dset_length can be a list.
#         """
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             f.require_group(group)
#             for i, dset_name in enumerate(dset_names):
#                 if type(dset_length) == list:
#                     assert len(dset_length) == len(
#                         dset_names
#                     ), "length of dset_length must match length of dset_names!"
#                     length = dset_length[i]
#                 else:
#                     length = dset_length
#                 f[group].create_dataset(dset_name, length)

#     def resize_hdf(self, dset_names, dset_length, group="/", file_name="data.h5"):
#         """
#         Resizes existing dataset in hdf5 file.
#         """
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             for dset_name in dset_names:
#                 f[group][dset_name].resize(dset_length)

#     def resize_group_hdf(
#         self, model, dset_length, group="model_weights", file_name="data.h5"
#     ):
#         assert group in ["model_weights", "info_quantities"]
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             for layer_name, layer in model.named_children():  # layer
#                 if group == "model_weights":
#                     for param_name, param in layer.named_parameters():
#                         for j in range(param.data.cpu().numpy().shape[0]):
#                             f[group][layer_name][str(j)][param_name].resize(
#                                 (dset_length, param.data.cpu().numpy()[j].size)
#                             )
#                 elif group == "info_quantities":
#                     for j in range(layer.output_size):
#                         f[group][layer_name][str(j)].resize(
#                             (dset_length, len(layer.info_quantities[j]))
#                         )

#     def write_group_dataset(
#         self, model, index, group="model_weights", file_name="data.h5"
#     ):
#         """
#         Writes model parameters to existing dataset in hdf5 file.
#         """
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             for layer_name, layer in model.named_children():  # layer
#                 f[group].require_group(layer_name)
#                 if group == "model_weights":
#                     for param_name, param in layer.named_parameters():
#                         for j in range(param.data.cpu().numpy().shape[0]):
#                             f[group][layer_name][str(j)][param_name][
#                                 index
#                             ] = param.data.cpu().numpy()[j]
#                 elif group == "info_quantities":
#                     for j in range(layer.output_size):
#                         f[group][layer_name][str(j)][index] = layer.info_quantities[j]

#     def write_to_dataset(self, dset_names, data, index, group="/", file_name="data.h5"):
#         """
#         Writes data to existing dataset in hdf5 file.
#         """
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             for dset_name, dset_data in zip(dset_names, data):
#                 dset_data = np.array(dset_data)
#                 if (
#                     dset_data.shape == () or dset_data.ndim == 2
#                 ):  # if data is a single number or a single element array
#                     f[group][dset_name][index] = dset_data
#                 else:
#                     # With this one can write batch-wise data
#                     idx = (index - 1) * dset_data.shape[
#                         0
#                     ] + 1  # -1 and +1 because we save initial performance at index 0
#                     f[group][dset_name][idx : idx + dset_data.shape[0]] = dset_data

#     def save_data(self, data, group="/", file_name="data.h5", **kwargs):
#         """
#         Saves dict to new dataset in hdf5 file. If data is not a numpy array, it is converted to one.
#         """
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             f.require_group(group)
#             for key in data:
#                 if type(data[key]).__module__ != np.__name__:
#                     data[key] = np.array(data[key])
#                 f[group].create_dataset(key, data=data[key], **kwargs)

#     def save_nested_data(self, data, group="/", file_name="data.h5", **kwargs):
#         """
#         Saves nested dict to new dataset in hdf5 file. If data is not a numpy array, it is converted to one.
#         """
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "a") as f:
#             f.require_group(group)
#             for key in data:
#                 f[group].require_group(key)
#                 for subkey in data[key]:
#                     if type(data[key][subkey]).__module__ != np.__name__:
#                         data[key][subkey] = np.array(data[key][subkey])
#                     f[group][key].create_dataset(
#                         subkey, data=data[key][subkey], **kwargs
#                     )

#     def save_model(self, model, epoch, optimizer, loss, seed):
#         cp_id = epoch
#         path = os.path.join(self.exp_directory, f"{self.cp_dir}/{cp_id}.pt")
#         if isinstance(optimizer, list):
#             optim_state = []
#             for opt in optimizer:
#                 optim_state.append(opt.state_dict())
#         else:
#             optim_state = optimizer.state_dict()
#         torch.save(
#             {
#                 "epoch": epoch,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optim_state,
#                 "loss": loss,
#                 "seed": seed,
#                 "model_class": model.__class__.__name__,
#             },
#             path,
#         )

#     def save_file(self, file_path, new_file_name="runfile.py"):
#         import shutil

#         shutil.copy(file_path, self.exp_directory + "/" + new_file_name)

#     # This part is only for analysis ===============================
#     def load_data_rec(self, group, skip=[], file_name="data.h5", path=None):
#         data = dict()
#         if path is None:
#             file = os.path.join(self.exp_directory, file_name)
#         else:
#             file = os.path.join(path, file_name)
#         with h5py.File(file, "r") as f:
#             keys = []
#             f[group].visit(
#                 lambda name: keys.append(name)
#                 if isinstance(f[group][name], h5py.Dataset)
#                 else None
#             )  # get all keys in group
#             for key in keys:
#                 if not any(s in key for s in skip):
#                     temp = data
#                     components = key.split("/")
#                     for out_key in components[:-1]:
#                         if out_key not in temp:
#                             temp[out_key] = dict()
#                         temp = temp[out_key]
#                     temp[components[-1]] = f[group][key][:]
#         return data

#     def load_data(self, dataset_name, file_name="data.h5"):
#         data = dict()
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "r") as f:
#             data["dataset_name"] = f[dataset_name][:]
#         return data

#     @staticmethod
#     def load_config(cfg_path, cfg_name="config.yaml"):
#         GlobalHydra.instance().clear()
#         if os.path.isabs(cfg_path):
#             initialize_config_dir(config_dir=cfg_path, version_base=None)
#         else:
#             initialize(config_path="../" + cfg_path, version_base=None)
        
#         overrides = []
#         if os.path.exists(os.path.join(cfg_path, 'overrides.yaml')):
#             overrides = OmegaConf.load(os.path.join(cfg_path, 'overrides.yaml'))
#         return compose(config_name=cfg_name, overrides=overrides)

#     def load_configs(self, add_run_properties=False, verbose=0):
#         df = None
#         for r, d, f in os.walk(self.exp_directory):
#             for file in f:
#                 try:
#                     if "config.yaml" == file:
#                         if os.path.isfile(os.path.join(r, "../../multirun.yaml")):
#                             multirun = 1
#                         else:
#                             multirun = 0
#                         path = os.path.join(r, file)
#                         if verbose > 0:
#                             print(path)
#                         time_dir = r.split("/")[-2 - multirun]
#                         date_dir = r.split("/")[-3 - multirun]
#                         cfg = yaml.safe_load(open(path))
#                         cfg["date_dir"] = date_dir
#                         cfg["time_dir"] = time_dir
#                         if multirun:
#                             cfg["multirun_id"] = r.split("/")[-2]
#                         else:
#                             cfg["multirun_id"] = "-1"
#                         cfg["run_path"] = r.replace(".hydra", "")
#                         cfg["conf_path"] = r
#                         if add_run_properties:
#                             run_properties = yaml.safe_load(
#                                 open(
#                                     os.path.join(cfg["run_path"], "run_properties.yaml")
#                                 )
#                             )
#                             # join the two dicts
#                             cfg = {**cfg, **run_properties}
#                         if df is None:
#                             df = pd.json_normalize(cfg)
#                         else:
#                             temp = pd.json_normalize(cfg)
#                             df = pd.concat([df, temp], axis=0)
#                         if verbose > 1:
#                             print(f"Loaded config from {path}")
#                 except Exception as e:
#                     if verbose > 0:
#                         print(f"Could not load config from {path}. Error: {e}")
#         df.sort_values(by=["date_dir", "time_dir"], inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         return df

#     def load_checkpoint(
#         self,
#         cfg,
#         run_path,
#         cp_id,
#         device="cpu",
#         runfile_name="runfile.py",
#         optimizer_names=None,
#     ):
#         path = os.path.join(f"{run_path}/{cfg.datamanager_params.cp_dir}/{cp_id}.pt")
#         checkpoint = torch.load(path)

#         model_cls = hf.load_class_from_file(
#             file_path=run_path + "/" + runfile_name,
#             class_name=checkpoint["model_class"],
#         )
#         model = model_cls(
#             cfg.layer_params, cfg.binning_params, cfg.optim_params, device
#         )
#         model.load_state_dict(checkpoint["model_state_dict"])
#         model.to(device)

#         opt = model.optimizers

#         if isinstance(checkpoint["optimizer_state_dict"], list):
#             for i, opt_state in enumerate(checkpoint["optimizer_state_dict"]):
#                 opt[i].load_state_dict(opt_state)
#         else:
#             opt.load_state_dict(checkpoint["optimizer_state_dict"])
#         return model, opt

#     def print_checkpoint_list(self, checkpoint_directory="checkpoints"):
#         files = os.listdir(os.path.join(self.exp_directory, checkpoint_directory))
#         files = [f for f in files if f.endswith(".pt")]
#         print(f"Checkpoint list: {files}")

#     def get_keys(self, group="/", file_name="data.h5"):
#         file = os.path.join(self.exp_directory, file_name)
#         with h5py.File(file, "r") as f:
#             return f[group].keys()

#     def load_data_of_group(
#         self, group, dataset_name="all", file_name="data.h5", path=None
#     ):
#         data = dict()
#         if path is None:
#             path = self.exp_directory
#         file = os.path.join(path, file_name)
#         if os.path.exists(file):
#             with h5py.File(file, "r") as f:
#                 if dataset_name == "all":
#                     keys = []
#                     f[group].visit(
#                         lambda name: keys.append(name)
#                         if isinstance(f[group][name], h5py.Dataset)
#                         else None
#                     )  # get all keys in group
#                     for key in f[group].keys():
#                         data[key] = f[group][key][:]
#                 else:
#                     data[dataset_name] = f[group][dataset_name][:]
#         else:
#             print(f"File {file} does not exist")

#         return data

#     def set_selected(self, sel):
#         self.sel = sel

#     def set_latest(self):
#         self.sel = self.all.iloc[-1:]

#     def load_selected(self, group, dataset_name="all", file_name="data.h5"):
#         data_all = [None] * len(self.sel)
#         self.sel.reset_index(drop=True, inplace=True)
#         for i, row in self.sel.iterrows():
#             data_all[i] = self.load_data_of_group(
#                 group, dataset_name, file_name, path=row["run_path"]
#             )
#         return data_all

#     def load_selected_dict(
#         self, group, dataset_name="all", file_name="data.h5", path=None
#     ):
#         data = dict()
#         if path is None:
#             path = self.exp_directory
#         file = os.path.join(path, file_name)

#         with h5py.File(file, "r") as f:
#             if dataset_name == "all":
#                 keys = []
#                 f[group].visit(
#                     lambda name: keys.append(name)
#                     if isinstance(f[group][name], h5py.Dataset)
#                     else None
#                 )  # get all keys in group
#                 for key in f[group].keys():
#                     data[key] = f[group][key][:]
#             else:
#                 data[dataset_name] = f[group][dataset_name][:]

#         return data

#     def load_selected_rec(self, group, skip=[], file_name="data.h5"):
#         data_all = [None] * len(self.sel)
#         self.sel.reset_index(drop=True, inplace=True)
#         for i, row in self.sel.iterrows():
#             data_all[i] = self.load_data_rec(
#                 group, skip, file_name, path=row["run_path"]
#             )
#         return data_all

#     def list_selected_datasets(self, group="/", file_name="data.h5"):
#         data_all = dict()
#         self.sel.reset_index(drop=True, inplace=True)
#         for i, row in self.sel.iterrows():
#             data_all[i] = self.list_file_datasets(
#                 group, file_name, path=row["run_path"]
#             )
#         return data_all

#     def list_file_datasets(self, group="/", file_name="data.h5", path=None):
#         file = os.path.join(path, file_name)
#         try:
#             with h5py.File(file, "r") as f:
#                 return list(f[group].keys())
#         except Exception as e:
#             print(f"Could not load file {file}. Error: {e}")
#             return []

class DataManager:
    def __init__(self, exp_directory=None, storage_config=None, cfg=None, mode='training', comment="No comment set.", add_run_properties=False, verbose=0):
        self.storage_config = storage_config
        self.exp_directory = exp_directory
        self.cfg = cfg

        if mode == 'training':
            self.checkpoints = self.genCheckpointArray()
            if storage_config.full_model: 
                os.makedirs(os.path.join(self.exp_directory, 'model_checkpoints'), exist_ok=True)
            self.genRunProperties(comment=comment)
        
        elif mode == 'analysis':
            self.all = self.load_configs(add_run_properties=add_run_properties, verbose=verbose)
            self.sel = copy.copy(self.all)  # select all by default

    def genCheckpointArray(self):
        number = self.storage_config.checkpoints.number
        spacing = self.storage_config.checkpoints.spacing
        if spacing == "log":
            checkpoints = np.unique(
                np.logspace(0, np.log10(self.cfg.exp_params.epochs + 1), number, dtype=int) - 1
            )  # log spaced checkpoints. Could be less than checkpoints
        elif spacing == "linear":
            checkpoints = np.unique(
                np.linspace(0, self.cfg.exp_params.epochs + 1, number, dtype=int, endpoint=True)
            )
        elif spacing == "all" and self.cfg.exp_params.epochs == number:
            checkpoints = np.arange(self.cfg.exp_params.epochs + 1)
        return checkpoints
    
    def genRunProperties(self, comment):
        user_email = subprocess.run(["git", "config", "user.email"], stdout=subprocess.PIPE).stdout.strip().decode()
        commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).stdout.strip().decode()
        properties = dict(user_email=user_email, commit_hash=commit_hash, finished=False, comment=comment,
            epochs_finished=0,)
        with open(os.path.join(self.exp_directory, "run_properties.yaml"), "w") as f:
            yaml.dump(properties, f)

    def init_hdf(self, model, num_batches):
        epochs = self.cfg.exp_params.epochs
        num_datapoints = (epochs+1)*num_batches if self.storage_config.batchwise else epochs+1

        if self.storage_config.performance:
            self.allocate_hdf(dset_names=["train_loss", "train_acc", "val_loss", "val_acc", "test_acc"],dset_length=[num_datapoints,num_datapoints,epochs+1, epochs+1, epochs+1,],group="performance")
            self.allocate_hdf(dset_names=['conf_matrix'], dset_length=(epochs+1, self.cfg.dataset.label_size**2), group='performance')

        if self.storage_config.pid:
            for layer_name, layer in model.named_children():  
                shape = (num_datapoints, hf.get_num_atoms(len(layer.input_sizes)), layer.output_size)
                self.allocate_hdf(dset_names=[layer_name], dset_length=shape, group="info_quantities",)

        if self.storage_config.model_params:
            self.allocate_weight_hdf(model, dset_length=num_datapoints)
        elif self.storage_config.final_model_params:
            self.allocate_weight_hdf(model, dset_length=1)

    def write_to_hdf(self, index, model=None, atoms=None, performances=None, optimizer=None):
        if self.storage_config.performance:
            self.write_to_dataset(dset_names=["train_loss", "train_acc", "val_loss", "val_acc", "test_acc",'conf_matrix'], data=performances, index=index, group='performance')

        if self.storage_config.pid:
            layer_names = [layer_name for layer_name in self.cfg.layer_params]
            self.write_to_dataset(dset_names=layer_names, data=atoms, index=index, group="info_quantities")

        if self.storage_config.full_model and index in self.checkpoints:
            self.save_model(model, index, optimizer, performances[0], self.cfg.exp_params.seed)

        if self.storage_config.model_params:
            self.write_group_dataset(model, index, group='model_weights')
        elif self.storage_config.final_model_params and index == self.cfg.exp_params.epochs:
            self.write_group_dataset(model, 0, group='model_weights')

        self.edit_run_properties(dict(epochs_finished=index))
        if index == self.cfg.exp_params.epochs:
            self.edit_run_properties(dict(finished=True))


    # Allocation functions
    def allocate_hdf(self, dset_names, dset_length, group="/", file_name="data.h5"):
        """
        Creates new dataset in hdf5 file. If the datasets are not of the same length, dset_length can be a list.
        """
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            f.require_group(group)
            for i, dset_name in enumerate(dset_names):
                if type(dset_length) == list:
                    assert len(dset_length) == len(dset_names), "length of dset_length must match length of dset_names!"
                    length = dset_length[i]
                else:
                    length = dset_length
                f[group].create_dataset(dset_name, length)

    def allocate_weight_hdf(self, model, dset_length, group="model_weights", file_name="data.h5"):
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            f.require_group(group)
            for layer_name, layer in model.named_children():  
                f[group].require_group(layer_name)
                for param_name, param in layer.named_parameters():
                    for j in range(param.data.cpu().numpy().shape[0]):
                        f[group][layer_name].require_group(str(j))
                        f[group][layer_name][str(j)].create_dataset(param_name,(dset_length, param.data.cpu().numpy()[j].size))

    # Writing functions
    def write_to_dataset(self, dset_names, data, index, group="/", file_name="data.h5"):
        """
        Writes data to existing dataset in hdf5 file.
        """
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            for dset_name, dset_data in zip(dset_names, data):
                dset_data = np.array(dset_data)
                if (dset_data.shape == () or dset_data.ndim == 2):  # if data is a single number or a single element array
                    f[group][dset_name][index] = dset_data
                else:
                    # With this one can write batch-wise data
                    idx = (index - 1) * dset_data.shape[0] + 1  # -1 and +1 because we save initial performance at index 0
                    f[group][dset_name][idx : idx + dset_data.shape[0]] = dset_data
    
    def write_group_dataset(self, model, index, group="model_weights", file_name="data.h5"):
        """
        Writes model parameters to existing dataset in hdf5 file.
        """
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            for layer_name, layer in model.named_children():  # layer
                f[group].require_group(layer_name)
                if group == "model_weights":
                    for param_name, param in layer.named_parameters():
                        for j in range(param.data.cpu().numpy().shape[0]):
                            f[group][layer_name][str(j)][param_name][index] = param.data.cpu().numpy()[j]

    def save_model(self, model: torch.nn.Module, index: int, optimizer: torch.optim.Optimizer|list, 
                   loss: float, seed: int):
        path = os.path.join(self.exp_directory, f"model_checkpoints/{index}.pt")
        if isinstance(optimizer, list):
            optim_state = []
            for opt in optimizer:
                optim_state.append(opt.state_dict())
        else:
            optim_state = optimizer.state_dict()

        torch.save({
                "epoch": index,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim_state,
                "loss": loss,
                "seed": seed,
                "model_class": model.__class__.__name__,
            },path)
        
    def edit_run_properties(self, changes: dict = None):
        with open(os.path.join(self.exp_directory, "run_properties.yaml"), "r") as f:
            properties = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in changes.items():
            properties[key] = value
        with open(os.path.join(self.exp_directory, "run_properties.yaml"), "w") as f:
            yaml.dump(properties, f)

    # Loading functions
    def load_configs(self, add_run_properties: bool = False, verbose: int = 0):
        configs = []
        for r, d, f in os.walk(self.exp_directory):
            for file in f:
                try:
                    if file == 'config.yaml':
                        if 'multirun' in d: multirun = 1
                        else: multirun = 0
                        path = os.path.join(r, file)
                        if verbose > 0: print(path)
                        with open(path) as f:
                            cfg = yaml.safe_load(f)
                        cfg["date_dir"] = r.split("/")[-3 - multirun]
                        cfg["time_dir"] = r.split("/")[-2 - multirun]
                        cfg["multirun_id"] = r.split("/")[-2] if multirun else "-1"
                        cfg["run_path"] = os.path.dirname(r)
                        
                        if add_run_properties:
                            with open(os.path.join(cfg["run_path"], "run_properties.yaml")) as f:
                                run_properties = yaml.safe_load(f)
                            cfg = {**cfg, **run_properties}

                        configs.append(cfg) 
                        if verbose > 1:
                            print(f"Successfully loaded config from {path}")
                except Exception as e:
                    if verbose > 0:
                        print(f"Could not load config from {path}. Error: {e}")

        df = pd.json_normalize(configs)
        df.sort_values(by=["date_dir", "time_dir"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def load_selected(self, group, dataset_name="all", file_name="data.h5"):
        data_all = [None] * len(self.sel)
        self.sel.reset_index(drop=True, inplace=True)
        for i, row in self.sel.iterrows():
            data_all[i] = self.load_data_of_group(
                group, dataset_name, file_name, path=row["run_path"]
            )
        return data_all

    def load_data_of_group(self, group, dataset_name="all", file_name="data.h5", path=None):
        data = dict()
        if path is None:
            path = self.exp_directory
        file = os.path.join(path, file_name)
        if os.path.exists(file):
            with h5py.File(file, "r") as f:
                if dataset_name == "all":
                    keys = []
                    f[group].visit(
                        lambda name: keys.append(name)
                        if isinstance(f[group][name], h5py.Dataset)
                        else None
                    )  # get all keys in group
                    for key in f[group].keys():
                        data[key] = f[group][key][:]
                else:
                    data[dataset_name] = f[group][dataset_name][:]
        else:
            print(f"File {file} does not exist")

        return data
    
    def save_file(self, file_path, new_file_name="runfile.py"):
        import shutil

        shutil.copy(file_path, self.exp_directory + "/" + new_file_name)