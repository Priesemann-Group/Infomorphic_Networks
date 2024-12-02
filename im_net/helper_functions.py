from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset
import importlib.util
import inspect

def confusion_mat(model, loader, use_taget=False):
    model.eval()
    with torch.no_grad():
        preds = None
        targets = None
        for x, target in loader:
            z = torch.nn.functional.one_hot(target).to(model.device)
            if not use_taget:
                z = torch.zeros_like(z).to(model.device)
            _, theta, _, _ = model(x.to(model.device), z)

            invert = torch.mean(theta, axis=0) > 0.5
            theta = theta ** (1 - invert.int()) * (1 - theta) ** invert.int()

            pred = theta.argmax(1)
            if preds is None:
                preds = pred
                targets = target

            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)

        cfm = confusion_matrix(preds.cpu(), targets.cpu())

        return cfm


def activation_map(activations, labels):
    n_classes = int(labels.max() + 1)
    n_samples = int(activations.shape[0])
    n_neur = int(activations.shape[1])

    act_map = torch.zeros((n_classes, n_neur))
    for i in range(n_classes):
        act_map[i, :] = torch.mean(activations[labels == i, :], axis=0)
    return act_map


def activation_hist(
    ax, activations, labels, bins=20, alpha=0.5, color=None, label=None
):
    n_classes = int(labels.max() + 1)
    n_samples = int(activations.shape[0])
    n_neur = int(activations.shape[1])

    if color is None:
        color = [None] * n_classes
    if label is None:
        label = [None] * n_classes

    ax.hist(activations.flatten(), bins=bins, alpha=alpha)
    ax.legend()


def acc_discrete(pred, target, check_inverse=True):
    acc = (pred == target).type(torch.get_default_dtype()).mean(axis=0)
    inv_acc = (pred == 1 - target).type(torch.get_default_dtype()).mean(axis=0)
    return torch.stack((acc, inv_acc)).max(axis=0)[0].mean().item()


def acc_continuous(pred, target, check_inverse=True, use_median=False):
    if use_median:
        med, _ = torch.median(pred.type(torch.get_default_dtype()), axis=0)
        invert = med > 0.5
    else:
        invert = torch.mean(pred.type(torch.get_default_dtype()), axis=0) > 0.5

    if check_inverse:
        pred_new = pred * (1 - invert.int()) + (1 - pred) * invert.int()
    else:
        pred_new = pred
    res = torch.argmax(pred_new, 1) == torch.argmax(target, 1)
    return torch.mean(res.type(torch.get_default_dtype())).item()


def acc_bartask(pred, target, correct_inverted=True, invert_others=False):
    if correct_inverted:
        pred_new = correct_inverse_neurons(pred, target, invert_others)
    else:
        pred_new = pred
    res = torch.mean(
        (((target + 1) / 2).int() == (pred_new > 0.5).int()).type(
            torch.get_default_dtype()
        )
    )
    return res


def correct_inverse_neurons(pred, invert_others=False):
    invert = torch.mean(pred, axis=0) < 0.5
    # print(pred, target, invert)
    if invert_others:
        invert = 1 - invert.int()
    pred_new = pred * (1 - invert.int()) + (1 - pred) * invert.int()
    return pred_new


def acc(pred, target):
    print("take care, this does not work likely")
    invert = torch.mean(pred.type(torch.get_default_dtype()), axis=0) > 0.5
    pred = pred ** (1 - invert.int()) * (1 - pred) ** invert.int()
    res = torch.argmax(pred, 1) == target
    return torch.mean(res.type(torch.get_default_dtype())).item()


def get_device(cuda_preference=True, job_id=0):
    if cuda_preference and torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        device_id = job_id % n_cuda
        device = torch.device("cuda:" + str(device_id))
    else:
        device = torch.device("cpu")
    return device

def load_class_from_file(file_path, class_name):
    file_name = file_path.split("/")[-1]
    spec = importlib.util.spec_from_file_location(file_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = None
    for member in inspect.getmembers(module):
        if member[0] == class_name:
            cls = member[1]
    return cls


def load_module(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def compute_error_signal(pred, target):
    for cl in range(pred.shape[1]):
        pred[:, cl] = correct_flipped(pred[:, cl], target[:, cl])
    return target - pred


def correct_flipped(pred, target):
    if (target == pred).type(torch.get_default_dtype()).mean().item() < (
        target == 1 - pred
    ).type(torch.get_default_dtype()).mean().item():
        return torch.ones_like(pred) - pred
    return pred

def get_num_atoms(num_sources):
    if num_sources == 2:
        return 5
    elif num_sources == 3:
        return 19
    else:
        raise NotImplementedError

def get_achains(num_sources, starting_idx=1):
    # defines the ordering of the gamma parameters throughout the code!
    # could be made more efficient by storing the chains in a dictionary
    if num_sources == 2:
        a_chains = [
            ((1,),),
            ((2,),),
            (
                (1,),
                (2,),
            ),
            ((1, 2),),
        ]
        if starting_idx != 1:
            a_chains = [
                tuple([tuple([i + starting_idx - 1 for i in tup]) for tup in chain])
                for chain in a_chains
            ]

    elif num_sources == 3:
        a_chains = [
            ((1,), (2,), (3,)),
            ((1,), (2,)),
            ((1,), (3,)),
            ((2,), (3,)),
            ((1,), (2, 3)),
            ((2,), (1, 3)),
            ((3,), (1, 2)),
            ((1,),),
            ((2,),),
            ((3,),),
            ((1, 2), (1, 3), (2, 3)),
            ((1, 2), (1, 3)),
            ((1, 2), (2, 3)),
            ((1, 3), (2, 3)),
            ((1, 2),),
            ((1, 3),),
            ((2, 3),),
            ((1, 2, 3),),
        ]
        if starting_idx != 1:
            a_chains = [
                tuple([tuple([i + starting_idx - 1 for i in tup]) for tup in chain])
                for chain in a_chains
            ]
    else:
        raise NotImplementedError
    return a_chains
