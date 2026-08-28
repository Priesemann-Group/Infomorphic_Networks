import copy
import os
import subprocess

import h5py
import im_net.helper_functions as hf
import numpy as np
import pandas as pd
import torch
import yaml
from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


class DataManager:
    def __init__(
        self,
        exp_directory,
        storage_config=None,
        cfg=None,
        mode="training",
        comment="No comment set.",
        add_run_properties=False,
        verbose=0,
    ):
        self.exp_directory = exp_directory
        self.storage_config = storage_config
        self.cfg = cfg
        self.verbose = verbose

        if mode == "training":
            if storage_config is not None and storage_config.full_model:
                self.checkpoints = self.genCheckpointArray()
                os.makedirs(os.path.join(self.exp_directory, "model_checkpoints"), exist_ok=True)
            self.genRunProperties(comment=comment)

        elif mode == "load":
            pass

        elif mode == "analysis":
            self.all = self.load_configs(add_run_properties=add_run_properties, verbose=verbose)
            self.sel = copy.copy(self.all)

    # ── Initialisation helpers ─────────────────────────────────────────────────

    def genCheckpointArray(self):
        """Returns an array of checkpoint epoch indices based on the storage config."""
        number = self.storage_config.checkpoints.number
        spacing = self.storage_config.checkpoints.spacing
        epochs = self.cfg.exp_params.epochs
        if spacing == "log":
            return np.unique(
                np.logspace(0, np.log10(epochs + 1), number, dtype=int) - 1
            )
        elif spacing == "linear":
            return np.unique(
                np.linspace(0, epochs + 1, number, dtype=int, endpoint=True)
            )
        elif spacing == "all":
            return np.arange(epochs + 1)

    def genRunProperties(self, comment=""):
        """Writes run_properties.yaml with git metadata to the experiment directory."""
        user_email = (
            subprocess.run(["git", "config", "user.email"], stdout=subprocess.PIPE)
            .stdout.strip().decode()
        )
        commit_hash = (
            subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
            .stdout.strip().decode()
        )
        properties = dict(
            user_email=user_email,
            commit_hash=commit_hash,
            finished=False,
            comment=comment,
            epochs_finished=0,
        )
        with open(os.path.join(self.exp_directory, "run_properties.yaml"), "w") as f:
            yaml.dump(properties, f)

    # ── High-level HDF5 interface ──────────────────────────────────────────────

    def init_hdf(self, model, num_batches):
        """Allocates all HDF5 datasets for a training run."""
        epochs = self.cfg.exp_params.epochs
        num_datapoints = (
            (epochs + 1) * num_batches if self.storage_config.batchwise else epochs + 1
        )
        if self.storage_config.performance:
            self.allocate_hdf(
                dset_names=["train_loss", "train_acc", "val_loss", "val_acc", "test_acc"],
                dset_length=[num_datapoints, num_datapoints, epochs + 1, epochs + 1, epochs + 1],
                group="performance",
            )
            self.allocate_hdf(
                dset_names=["conf_matrix"],
                dset_length=(epochs + 1, self.cfg.dataset.label_size ** 2),
                group="performance",
            )
        if self.storage_config.pid:
            for layer_name, layer in model.named_children():
                shape = (num_datapoints, hf.get_num_atoms(len(layer.input_sizes)), layer.output_size)
                self.allocate_hdf(dset_names=[layer_name], dset_length=shape, group="info_quantities")
        if self.storage_config.model_params:
            self.allocate_weight_hdf(model, dset_length=num_datapoints)
        elif self.storage_config.final_model_params:
            self.allocate_weight_hdf(model, dset_length=1)

    def write_to_hdf(self, index, model=None, atoms=None, performances=None, optimizer=None):
        """Writes all tracked quantities for one epoch/step to HDF5 and checkpoints."""
        if self.storage_config.performance:
            self.write_to_dataset(
                dset_names=["train_loss", "train_acc", "val_loss", "val_acc", "test_acc", "conf_matrix"],
                data=performances,
                index=index,
                group="performance",
            )
        if self.storage_config.pid:
            self.write_to_dataset(
                dset_names=list(self.cfg.layer_params),
                data=atoms,
                index=index,
                group="info_quantities",
            )
        if self.storage_config.full_model and index in self.checkpoints:
            self.save_model(model, index, optimizer, performances[0], self.cfg.exp_params.seed)
        if self.storage_config.model_params:
            self.write_group_dataset(model, index)
        elif self.storage_config.final_model_params and index == self.cfg.exp_params.epochs:
            self.write_group_dataset(model, 0)

        self.edit_run_properties({"epochs_finished": index})
        if index == self.cfg.exp_params.epochs:
            self.edit_run_properties({"finished": True})

    # ── Allocation ─────────────────────────────────────────────────────────────

    def allocate_hdf(self, dset_names, dset_length, group="/", file_name="data.h5"):
        """Creates new datasets in an HDF5 file.

        ``dset_length`` may be a list (one length per dataset) or a single
        value applied to all datasets.
        """
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            f.require_group(group)
            for i, dset_name in enumerate(dset_names):
                length = dset_length[i] if isinstance(dset_length, list) else dset_length
                f[group].create_dataset(dset_name, length, dtype='f4')

    def allocate_weight_hdf(self, model, dset_length, group="model_weights", file_name="data.h5"):
        """Allocates per-neuron, per-parameter HDF5 datasets for model weights."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            f.require_group(group)
            for layer_name, layer in model.named_children():
                f[group].require_group(layer_name)
                for param_name, param in layer.named_parameters():
                    param_np = param.data.cpu().numpy()          # computed once per param
                    for j in range(param_np.shape[0]):
                        f[group][layer_name].require_group(str(j))
                        f[group][layer_name][str(j)].create_dataset(
                            param_name, (dset_length, param_np[j].size), dtype='f4'
                        )

    def resize_hdf(self, dset_names, dset_length, group="/", file_name="data.h5"):
        """Resizes existing datasets in an HDF5 file."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            for dset_name in dset_names:
                f[group][dset_name].resize(dset_length)

    def resize_group_hdf(self, model, dset_length, group="model_weights", file_name="data.h5"):
        """Resizes existing per-parameter HDF5 datasets for model weights or info quantities."""
        assert group in ["model_weights", "info_quantities"]
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            for layer_name, layer in model.named_children():
                if group == "model_weights":
                    for param_name, param in layer.named_parameters():
                        param_np = param.data.cpu().numpy()      # computed once per param
                        for j in range(param_np.shape[0]):
                            f[group][layer_name][str(j)][param_name].resize(
                                (dset_length, param_np[j].size)
                            )
                elif group == "info_quantities":
                    for j in range(layer.output_size):
                        f[group][layer_name][str(j)].resize(
                            (dset_length, len(layer.info_quantities[j]))
                        )

    # ── Writing ────────────────────────────────────────────────────────────────

    def write_to_dataset(self, dset_names, data, index, group="/", file_name="data.h5"):
        """Writes data to existing datasets in an HDF5 file.

        Scalars and 2-D arrays are written at ``index``; 1-D arrays are
        treated as batches and written into a contiguous slice.
        """
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            for dset_name, dset_data in zip(dset_names, data):
                dset_data = np.array(dset_data)
                if dset_data.shape == () or dset_data.ndim == 2:
                    f[group][dset_name][index] = dset_data
                else:
                    # Batch-wise: index 0 holds the initial value
                    idx = (index - 1) * dset_data.shape[0] + 1
                    f[group][dset_name][idx: idx + dset_data.shape[0]] = dset_data

    def write_group_dataset(self, model, index, group="model_weights", file_name="data.h5"):
        """Writes model parameters to existing HDF5 datasets."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            for layer_name, layer in model.named_children():
                f[group].require_group(layer_name)
                for param_name, param in layer.named_parameters():
                    param_np = param.data.cpu().numpy()          # computed once per param
                    for j in range(param_np.shape[0]):
                        f[group][layer_name][str(j)][param_name][index] = param_np[j]

    def save_data(self, data, group="/", file_name="data.h5", **kwargs):
        """Saves a flat dict to new datasets in an HDF5 file."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            f.require_group(group)
            for key, value in data.items():
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                f[group].create_dataset(key, data=value, **kwargs)

    def save_nested_data(self, data, group="/", file_name="data.h5", **kwargs):
        """Saves a nested dict to new datasets in an HDF5 file."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "a") as f:
            f.require_group(group)
            for key, subdict in data.items():
                f[group].require_group(key)
                for subkey, value in subdict.items():
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)
                    f[group][key].create_dataset(subkey, data=value, **kwargs)

    def save_model(
        self,
        model: torch.nn.Module,
        index: int,
        optimizer: torch.optim.Optimizer | list,
        loss: float,
        seed: int,
    ):
        """Saves a model checkpoint to ``model_checkpoints/<index>.pt``."""
        path = os.path.join(self.exp_directory, f"model_checkpoints/{index}.pt")
        optim_state = (
            [opt.state_dict() for opt in optimizer]
            if isinstance(optimizer, list)
            else optimizer.state_dict()
        )
        torch.save(
            {
                "epoch": index,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim_state,
                "loss": loss,
                "seed": seed,
                "model_class": model.__class__.__name__,
            },
            path,
        )

    def save_file(self, file_path, new_file_name="runfile.py"):
        """Copies a file into the experiment directory."""
        import shutil
        shutil.copy(file_path, os.path.join(self.exp_directory, new_file_name))

    # ── Run properties ─────────────────────────────────────────────────────────

    def edit_run_properties(self, changes: dict = None):
        """Updates fields in run_properties.yaml."""
        path = os.path.join(self.exp_directory, "run_properties.yaml")
        with open(path, "r") as f:
            properties = yaml.load(f, Loader=yaml.FullLoader)
        properties.update(changes)
        with open(path, "w") as f:
            yaml.dump(properties, f)

    def rerun_failed_runs(self, num_cpus=2):
        failed_runs = self.all[self.all.finished == False]
        raise NotImplementedError

    # ── Loading ────────────────────────────────────────────────────────────────

    @staticmethod
    def load_config(cfg_path, cfg_name="config.yaml"):
        """Loads a single Hydra config from ``cfg_path``."""
        GlobalHydra.instance().clear()
        if os.path.isabs(cfg_path):
            initialize_config_dir(config_dir=cfg_path, version_base=None)
        else:
            initialize(config_path="../" + cfg_path, version_base=None)
        overrides = []
        if os.path.exists(os.path.join(cfg_path, "overrides.yaml")):
            overrides = OmegaConf.load(os.path.join(cfg_path, "overrides.yaml"))
        return compose(config_name=cfg_name, overrides=overrides)

    def load_configs(self, add_run_properties: bool = False, verbose: int = 0):
        """Walks the experiment directory and returns a DataFrame of all run configs."""
        configs = []
        for r, d, f in os.walk(self.exp_directory):
            for file in f:
                try:
                    if file == "config.yaml":
                        multirun = 1 if "multirun" in d else 0
                        path = os.path.join(r, file)
                        if verbose > 0:
                            print(path)
                        with open(path) as fh:
                            cfg = yaml.safe_load(fh)
                        cfg["date_dir"] = r.split("/")[-3 - multirun]
                        cfg["time_dir"] = r.split("/")[-2 - multirun]
                        cfg["multirun_id"] = r.split("/")[-2] if multirun else "-1"
                        cfg["run_path"] = os.path.dirname(r)
                        if add_run_properties:
                            with open(os.path.join(cfg["run_path"], "run_properties.yaml")) as fh:
                                run_properties = yaml.safe_load(fh)
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

    def load_checkpoint(
        self,
        cfg,
        run_path,
        cp_id,
        device="cpu",
        runfile_name="runfile.py",
        optimizer_names=None,
    ):
        """Loads a model and its optimizer(s) from a checkpoint."""
        path = f"{run_path}/{cfg.datamanager_params.cp_dir}/{cp_id}.pt"
        checkpoint = torch.load(path)
        model_cls = hf.load_class_from_file(
            file_path=f"{run_path}/{runfile_name}",
            class_name=checkpoint["model_class"],
        )
        model = model_cls(cfg.layer_params, cfg.binning_params, cfg.optim_params, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        opt = model.optimizers
        if isinstance(checkpoint["optimizer_state_dict"], list):
            for i, opt_state in enumerate(checkpoint["optimizer_state_dict"]):
                opt[i].load_state_dict(opt_state)
        else:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, opt

    def load_checkpoint_IM(
        self,
        cfg,
        run_path,
        cp_id,
        device="cpu",
        runfile_name="runfile.py",
        optimizer_names=None,
    ):
        """Loads an IM model from a checkpoint (no optimizer state returned)."""
        path = f"{run_path}/{cfg.datamanager_params.cp_dir}/{cp_id}.pt"
        checkpoint = torch.load(path)
        binning_cls = hf.load_module(cfg.binning_params.name)
        binning_method = binning_cls(device, **cfg.binning_params.params)
        model_cls = hf.load_class_from_file(
            file_path=f"{run_path}/{runfile_name}",
            class_name=checkpoint["model_class"],
        )
        model = model_cls(cfg.layer_params, binning_method)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model

    def load_data(self, dataset_name, file_name="data.h5"):
        """Loads a single top-level dataset from an HDF5 file."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "r") as f:
            return {dataset_name: f[dataset_name][:]}

    def load_data_of_group(self, group, dataset_name="all", file_name="data.h5", path=None):
        """Loads top-level datasets from an HDF5 group into a dict."""
        if path is None:
            path = self.exp_directory
        file = os.path.join(path, file_name)
        if not os.path.exists(file):
            print(f"File {file} does not exist")
            return {}
        with h5py.File(file, "r") as f:
            if dataset_name == "all":
                return {key: f[group][key][:] for key in f[group].keys()}
            return {dataset_name: f[group][dataset_name][:]}

    def load_data_rec(self, group, skip=[], file_name="data.h5", path=None):
        """Recursively loads all datasets in an HDF5 group into a nested dict."""
        file = os.path.join(path if path is not None else self.exp_directory, file_name)
        data = {}
        with h5py.File(file, "r") as f:
            dataset_keys = []
            f[group].visit(
                lambda name: dataset_keys.append(name)
                if isinstance(f[group][name], h5py.Dataset)
                else None
            )
            for key in dataset_keys:
                if any(s in key for s in skip):
                    continue
                *parts, leaf = key.split("/")
                node = data
                for part in parts:
                    node = node.setdefault(part, {})
                node[leaf] = f[group][key][:]
        return data

    def load_selected_dict(self, group, dataset_name="all", file_name="data.h5", path=None):
        """Loads datasets from an HDF5 group into a dict (single path variant)."""
        if path is None:
            path = self.exp_directory
        return self.load_data_of_group(group, dataset_name, file_name, path=path)

    # ── Selection helpers ──────────────────────────────────────────────────────

    def set_selected(self, sel):
        self.sel = sel

    def set_latest(self):
        self.sel = self.all.iloc[-1:]

    def load_selected(self, group, dataset_name="all", file_name="data.h5"):
        """Loads group data for every currently selected run."""
        self.sel.reset_index(drop=True, inplace=True)
        return [
            self.load_data_of_group(group, dataset_name, file_name, path=row["run_path"])
            for _, row in self.sel.iterrows()
        ]

    def load_selected_rec(self, group, skip=[], file_name="data.h5"):
        """Recursively loads group data for every currently selected run."""
        self.sel.reset_index(drop=True, inplace=True)
        return [
            self.load_data_rec(group, skip, file_name, path=row["run_path"])
            for _, row in self.sel.iterrows()
        ]

    # ── Listing helpers ────────────────────────────────────────────────────────

    def print_checkpoint_list(self, checkpoint_directory="checkpoints"):
        files = os.listdir(os.path.join(self.exp_directory, checkpoint_directory))
        print("Checkpoint list:", [f for f in files if f.endswith(".pt")])

    def get_keys(self, group="/", file_name="data.h5"):
        """Returns the top-level keys of an HDF5 group."""
        file = os.path.join(self.exp_directory, file_name)
        with h5py.File(file, "r") as f:
            return list(f[group].keys())

    def list_selected_datasets(self, group="/", file_name="data.h5"):
        """Returns a dict mapping run index → dataset names for all selected runs."""
        self.sel.reset_index(drop=True, inplace=True)
        return {
            i: self.list_file_datasets(group, file_name, path=row["run_path"])
            for i, row in self.sel.iterrows()
        }

    def list_file_datasets(self, group="/", file_name="data.h5", path=None):
        """Returns the top-level dataset names in an HDF5 group."""
        file = os.path.join(path, file_name)
        try:
            with h5py.File(file, "r") as f:
                return list(f[group].keys())
        except Exception as e:
            print(f"Could not load file {file}. Error: {e}")
            return []