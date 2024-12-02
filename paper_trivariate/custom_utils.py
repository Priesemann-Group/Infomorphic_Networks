from hydra.core.hydra_config import HydraConfig
from im_net import datamanager, datasets
from im_net import helper_functions as hf
import matplotlib.pyplot as plt
import torch
import os

# def init_exp_dir(datamanager_params, epochs, require_comment_singlerun=True, require_comment_multirun=True):
#     """
#     Initialize the experiment directory via the datamanager.
#     Asks for a comment to be saved in the run_properties.yaml file - this can be used to describe the run.
#     """
#     comment = ''
#     if str(HydraConfig.get().mode) == "RunMode.RUN": # single run
#         exp_directory = HydraConfig.get().run.dir
#         if require_comment_singlerun: comment = input('Comment to run: ')

#     else: # multirun
#         exp_directory = os.path.join(HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir)
#         if require_comment_multirun:
#             if HydraConfig.get().job.num == 0:
#                 comment = input('Comment to run: ')
#             else:
#                 comment = 'See job number 0'
    
#     # create a datamanager instance
#     dm = datamanager.DataManager(exp_directory=exp_directory, **datamanager_params, epochs=epochs)
#     dm.edit_run_properties(dict(comment=comment))

#     # save this runfile to experiment directory
#     dm.save_file(__file__)
#     return dm

def init_exp_dir(storage_config, config, runfile):
    """
    Initialize the experiment directory via the datamanager.
    Asks for a comment to be saved in the run_properties.yaml file - this can be used to describe the run.
    """
    comment = ''
    if str(HydraConfig.get().mode) == "RunMode.RUN": # single run
        directory = HydraConfig.get().run.dir
        if storage_config.comment: comment = input('Comment to run: ')

    else: # multirun
        directory = os.path.join(HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir)
        if storage_config.comment:
            if HydraConfig.get().job.num == 0:
                comment = input('Comment to multirun: ')
            else:
                comment = 'See job number 0'
    
    # create a datamanager instance
    dm = datamanager.DataManager(exp_directory=directory, storage_config=storage_config, cfg=config, comment=comment)

    # save this runfile to experiment directory
    dm.save_file(runfile) # currently not working
    return dm

def plot_conf_matrix(conf_matrix):
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(5,5), dpi=100)
    ax.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{conf_matrix[i,j]:.2f}', ha='center', va='center', color='w')
    plt.tight_layout()
    plt.savefig(os.path.join("./", 'conf_matrix.png'))
    plt.close()

def get_device(pref_gpu):
    """
    Initialize the device to be used for training.
    Automatically chooses GPU if available, otherwise uses CPU.
    If multiple GPUs are available, the job_id is used to select one.
    """
    if str(HydraConfig.get().mode) == "RunMode.RUN": # single run
        job_id = 0
    else: # multirun
        job_id = HydraConfig.get().job.num
    device = hf.get_device(cuda_preference=pref_gpu, job_id=job_id)
    return device


def init_dataloaders(dataset_cfg, exp_params, data_dir='../datasets/', device=None):
    """
    Initialize the dataloaders for training, validation and testing.
    """
    trainset, testset = datasets.grab_data(data_dir, dataset=dataset_cfg.name, creation_seed=exp_params.seed, device=device, **dataset_cfg.generation if 'generation' in dataset_cfg else {})
    trainset, valset = datasets.generate_train_val_data_split(trainset, exp_params.seed, dataset_cfg.params.frac_val)
    trainloader, valloader, testloader = datasets.init_data_loaders(trainset, valset, testset, exp_params.batch_size)
    return trainloader, valloader, testloader

def gen_sparse_connectivity(n_in, n_out, max_con):
    if max_con < n_in:
        con = torch.zeros((n_out, n_in))
        for i in range(n_out):
            without_self = torch.zeros(n_in-1)
            without_self[torch.randperm(n_in-1)[:max_con]] = 1
            con[i] = torch.cat((without_self[:i], torch.tensor([0]), without_self[i:]))
        con = torch.clamp((con-torch.eye(n_out)),0,1)
    else:
        if n_in == n_out:
            con = 1-torch.eye(n_out)
        else:
            raise ValueError('Not implemented')
    return con