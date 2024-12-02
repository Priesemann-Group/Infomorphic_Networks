import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import random
import copy
import warnings
from itertools import product
import torch.nn.functional as F


def get_normalization_transform(trainset):
    num_samples = trainset.data.shape[0]
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=num_samples
    )
    imgs, _ = next(iter(trainloader))
    dataset_mean = torch.stack([x.mean() for x, _ in trainloader]).mean()
    dataset_std = torch.stack([x.std() for x, _ in trainloader]).mean()

    normalized_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ]
    )
    return normalized_transform


def grab_data(data_dir, dataset="CIFAR10", creation_seed=None, device=None, **kwargs):
    """Downloads train and test set, stores them on disk, computes mean
        and standard deviation per channel of trainset, normalizes the train set
        accordingly.

    Args:
        data_dir (str): Directory to store data
        dataset(str): Name of the dataset to load

    Returns:
        dataset, dataset: Returns trainset and testset as
            torchvision dataset objects.
    """
    custom_datasets = {
        "Gaussian": GaussianDataset,
        "BarsHV": BarsHVDataset,
        "RandomPattern": RandomPatternDataset,
        "Olshausen": OlshausenDataset,
        "BinaryPredictionDataset": BinaryPredictionDataset,
        "MultiICA": MultiIcaDataset
    }
    if dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        # Get normalization transformation
        normalized_transform = get_normalization_transform(
            trainset
        )

        # Load again, now normalized
        trainset = torchvision.datasets.CIFAR10(
            data_dir, download=True, train=True, transform=normalized_transform
        )
        testset = torchvision.datasets.CIFAR10(
            data_dir, download=True, train=False, transform=normalized_transform
        )
        return trainset, testset

    if dataset == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(
            data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        # Get normalization transform
        normalized_transform = get_normalization_transform(
            trainset
        )

        # Load again, now normalized
        trainset = torchvision.datasets.CIFAR10(
            data_dir, download=True, train=True, transform=normalized_transform
        )
        testset = torchvision.datasets.CIFAR10(
            data_dir, download=True, train=False, transform=normalized_transform
        )
        return trainset, testset
    
    if dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        testset = torchvision.datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        train_images = trainset.data.float()
        train_labels = trainset.targets

        test_images = testset.data.float()
        test_labels = testset.targets

        # Normalize images
        mean = train_images.mean()
        std = train_images.std()

        train_images = (train_images - mean) / std
        test_images = (testset.data - mean) / std

        # Move to GPU
        if device is not None:
            train_images = train_images.to(device)
            test_images = test_images.to(device)
            train_labels = train_labels.to(device)
            test_labels = test_labels.to(device)

        trainset = torch.utils.data.TensorDataset(train_images, train_labels)
        testset = torch.utils.data.TensorDataset(test_images, test_labels)

        return trainset, testset

    if dataset == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(
            data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        # Get normalization transform
        normalized_transform = get_normalization_transform(
            trainset
        )

        # Load again, now normalized
        trainset = torchvision.datasets.FashionMNIST(
            data_dir, download=True, train=True, transform=normalized_transform
        )
        testset = torchvision.datasets.FashionMNIST(
            data_dir, download=True, train=False, transform=normalized_transform
        )
        return trainset, testset
    
    if dataset == "RHM":
        dataset = RandomHierarchyModel(**kwargs)
        trainset = torch.utils.data.Subset(dataset, range(kwargs['train_size']))
        testset = torch.utils.data.Subset(dataset, range(kwargs["train_size"], kwargs["train_size"] + kwargs["test_size"]))
        
        return trainset, testset
    
    if dataset in custom_datasets.keys():
        return dataset_helper(
            custom_datasets[dataset], creation_seed, **kwargs
        )  # returns a tuple of datasets
    
    raise ValueError("Dataset not implemented")


def dataset_helper(dataset_cls, creation_seed=None, **kwargs):
    """Helper function to create multiple datasets of the same type."""
    # set seed
    if "creation_seed" != None:
        torch.manual_seed(creation_seed)
    datasets = []
    if "num_samples" not in kwargs:
        datasets.append(dataset_cls(**kwargs))
    else:
        for n in list(kwargs["num_samples"]):
            kwargs["num_samples"] = n
            datasets.append(dataset_cls(**kwargs))
    return tuple(datasets)


def generate_train_val_data_split(trainset, split_seed=42, val_frac=0.2):
    """Splits train dataset into train and validation dataset.

    Args:
        trainset (CIFAR10): CIFAR10 trainset object
        split_seed (int, optional): Seed used to randomly assign data
            points to the validation set. Defaults to 42.
        val_frac (float, optional): Fraction of training set that should be
            split into validation set. Defaults to 0.2.

    Returns:
        CIFAR10, CIFAR10: CIFAR10 trainset and validation set.
    """
    num_val_samples = np.ceil(val_frac * len(trainset)).astype(int)
    num_train_samples = len(trainset) - num_val_samples

    generator =torch.Generator() if split_seed is None else torch.Generator().manual_seed(split_seed)
    trainset, valset = torch.utils.data.random_split(
        trainset,
        (num_train_samples, num_val_samples),
        generator=generator,
    )

    return trainset, valset


def init_data_loaders(trainset=None, valset=None, testset=None, batch_size=32):
    """Initialize train, validation and test data loader.

    Args:
        trainset (CIFAR10): Training set torchvision dataset object.
        valset (CIFAR10): Validation set torchvision dataset object.
        testset (CIFAR10): Test set torchvision dataset object.
        batch_size (int, optional): Batchsize that should be generated by
            pytorch dataloader object. Defaults to 1024.

    Returns:
        DataLoader, DataLoader, DataLoader: Returns pytorch DataLoader objects
            for training, validation and testing.
    """
    loaders = []
    if trainset is not None:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        loaders.append(trainloader)
    if valset is not None:
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True
        )
        loaders.append(valloader)
    if testset is not None:
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True
        )
        loaders.append(testloader)
    return loaders


class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BinaryPredictionDataset(CustomDataset):
    def __init__(self, num_samples, **kwargs):
        a = kwargs["fire_prob"] * torch.ones(num_samples)
        self.labels = torch.bernoulli(a)
        self.data = torch.bernoulli(torch.abs(self.labels - kwargs["wrong_fraction"]))
        if kwargs["flip"]:
            self.data = torch.abs(self.data - 1)
        samples = list(zip(self.data, self.labels))
        super().__init__(samples)


class GaussianDataset(CustomDataset):
    def __init__(self, num_samples, num_classes=None, std=0.3, dim=1, **kwargs):
        assert dim == 1, "Only 1D Gaussian implemented"
        if kwargs["classes"] is not None:
            num_classes = len(kwargs["classes"])
            rand = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
            self.labels = torch.tensor(
                kwargs["classes"], dtype=torch.get_default_dtype()
            )[rand]
        else:
            assert (
                num_classes is not None
            ), "Either classes or num_classes must be specified"
            self.labels = torch.randint(
                -1, num_classes - 1, (num_samples,), dtype=torch.get_default_dtype()
            )
        self.data = torch.normal(self.labels, std)
        samples = list(zip(self.data, self.labels))
        super().__init__(samples)


class BarsHVDataset(CustomDataset):
    def __init__(
        self,
        num_samples=10000,
        size=8,
        horizontal=True,
        vertical=False,
        p=0.5,
        min=-1,
        **kwargs
    ):
        patterns = torch.zeros((num_samples, size, size))
        if horizontal:
            h_on = 1.0 * (torch.rand(num_samples, size) < p)
            h_bars = np.repeat(h_on[:, :, np.newaxis], size, axis=2)
            patterns += h_bars
        if vertical:
            v_on = 1.0 * (torch.rand(num_samples, size) < p)
            v_bars = np.repeat(v_on[:, np.newaxis, :], size, axis=1)
            patterns += v_bars
        if min == 0:
            self.data = patterns
        else:
            self.data = 2 * torch.clip(patterns, 0, 1) - 1
        super().__init__(self.data)


class RandomPatternDataset(CustomDataset):
    def __init__(self, num_samples=10000, size=8, p=0.5, **kwargs):
        self.data = torch.randint(0, 2, (num_samples, size, size)) * 2.0 - 1.0
        super().__init__(self.data)


class OlshausenDataset(CustomDataset):
    def __init__(
        self,
        data_dir,
        num_samples=10000,
        size=8,
        raw=False,
        clip_and_rescale=None,
        **kwargs
    ):
        from scipy.io import loadmat
        import os

        if raw:
            images = loadmat(
                os.path.join(data_dir, "olshausen/IMAGES_RAW.mat"),
                variable_names="IMAGESr",
                appendmat=True,
            ).get("IMAGESr")
        else:
            images = loadmat(
                os.path.join(data_dir, "olshausen/IMAGES.mat"),
                variable_names="IMAGES",
                appendmat=True,
            ).get("IMAGES")

        h_full, w_full, num_images = images.shape

        w, h = size, size
        h_start = np.random.randint(0, h_full - h, size=(num_samples,))
        w_start = np.random.randint(0, w_full - w, size=(num_samples,))
        rnd_image = np.random.randint(0, num_images, size=(num_samples,))

        patterns = np.zeros((num_samples, h, w))

        for i in range(num_samples):
            patterns[i] = images[
                h_start[i] : h_start[i] + h, w_start[i] : w_start[i] + w, rnd_image[i]
            ]

        if clip_and_rescale != None:
            patterns = np.clip(patterns, -clip_and_rescale, clip_and_rescale)
            patterns = (1.0 / clip_and_rescale) * patterns
        self.data = torch.tensor(patterns).type(torch.get_default_dtype())
        super().__init__(self.data)


class MultiIcaDataset(CustomDataset):
    def __init__(
        self,
        num_samples=10000,
        dim=2,
        pdf="np.random.triangular",
        transform_mat=[[1, 0], [0, 1]],
        **kwargs
    ):
        pdf_func = eval(pdf)(**kwargs["pdf_kwargs"], size=(num_samples, dim))
        self.data = torch.tensor(pdf_func).type(torch.get_default_dtype())
        self.data = torch.matmul(
            self.data, torch.tensor(transform_mat).type(torch.get_default_dtype())
        )
        super().__init__(self.data)


class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,
            num_classes=2,
            num_synonyms=2, #
            tuple_size=2,	# size of the low-level representations
            num_layers=2,
            seed_rules=0,
            seed_sample=1,
            train_size=-1,
            test_size=0,
            input_format='onehot',
            whitening=0,
            transform=None,
            replacement=False,
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms 
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size

        rules = sample_rules( num_features, num_classes, num_synonyms, tuple_size, num_layers, seed=seed_rules)
        self.rules = rules
 
        max_data = num_classes * num_synonyms ** ((tuple_size ** num_layers - 1) // (tuple_size - 1)) #
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > 1e19 and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:
            self.features, self.labels = sample_without_replacement(
                max_data, train_size, test_size, seed_sample, rules
            )
        else:
            self.features, self.labels = sample_with_replacement(
                train_size, test_size, seed_sample, rules
            )

        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

	# TODO: implement one-hot encoding of s-tuples
        if 'onehot' in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features if 'tuples' not in input_format else num_features ** tuple_size
            ).float()
            
            if whitening:

                inv_sqrt_norm = (1.-1./num_features) ** -.5
                self.features = (self.features - 1./num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)
            self.features = self.features.reshape(self.features.shape[0], -1)
        elif 'long' in input_format:
            self.features = self.features.long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

    def get_rules(self):
        return self.rules    
    

def dec2bin(n, bits=None):
    """
    Convert integers to binary.
    
    Args:
            n: The numbers to convert (tensor of size [*]).
         bits: The length of the representation.
    Returns:
        A tensor (size [*, bits]) with the binary representations.
    """
    if bits is None:
        bits = (x.max() + 1).log2().ceil().item()
    x = x.int()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def dec2base(n, b, length=None):
    """
    Convert integers into a different base.
    
    Args:
            n: The numbers to convert (tensor of size [*]).
            b: The base (integer).
       length: The length of the representation.
    Returns:
        A tensor (size [*, length]) containing the input numbers in the new base.
    """
    digits = []
    while n.sum():
        digits.append(n % b)
        n = n.div(b, rounding_mode='floor')
    if length:
        assert len(digits) <= length, "Length required is too small to represent input numbers!"
        digits += [torch.zeros(len(n), dtype=int)] * (length - len(digits))
    return torch.stack(digits[::-1]).t()

def sample_rules( v, n, m, s, L, seed=42):
        """
        Sample random rules for a random hierarchy model.

        Args:
            v: The number of values each variable can take (vocabulary size, int).
            n: The number of classes (int).
            m: The number of synonymic lower-level representations (multiplicity, int).
            s: The size of lower-level representations (int).
            L: The number of levels in the hierarchy (int).
            seed: Seed for generating the rules.

        Returns:
            A dictionary containing the rules for each level of the hierarchy.
        """
        random.seed(seed)
        tuples = list(product(*[range(v) for _ in range(s)]))

        rules = {}
        rules[0] = torch.tensor(
                random.sample( tuples, n*m)
        ).reshape(n,m,-1)
        for i in range(1, L):
            rules[i] = torch.tensor(
                    random.sample( tuples, v*m)
            ).reshape(v,m,-1)

        return rules


def sample_data_from_generator_classes(g, y, rules, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from its rules, a seed and a set of class labels.

    Args:
        g: A torch.Generator object.
        y: A tensor of size [batch_size, 1] containing the class labels.
        rules: A dictionary containing the rules for each level of the hierarchy.
        return_tree_structure: If True, return the tree structure of the hierarchy as a dictionary.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    labels = copy.deepcopy(y)

    if return_tree_structure:
        x_st = (
            {}
        )  # Initialize the dictionary to store the hidden variables
        x_st[0] = y
        for i in range(L):  # Loop over the levels of the hierarchy
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=g
            )  # Choose a random rule for each variable in the current level
            x_st[i + 1] = rules[i][x_st[i], chosen_rule].flatten(
                start_dim=1
            )  # Apply the chosen rule to each variable in the current level
        return x_st, labels
    else:
        x = y
        for i in range(L):
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x.shape, generator=g
            )
            x = rules[i][x, chosen_rule].flatten(start_dim=1)
        return x, labels
    

def sample_with_replacement(train_size, test_size, seed_sample, rules):

    n = rules[0].shape[0]  # Number of classes

    if train_size == -1:
        warnings.warn(
            "Whole dataset (train_size=-1) not available with replacement! Using train_size=1e6.",
            RuntimeWarning,
        )
        train_size = 1000000

    g = torch.Generator()
    g.manual_seed(seed_sample)

    y = torch.randint(low=0, high=n, size=(train_size + test_size,), generator=g)
    features, labels = sample_data_from_generator_classes(g, y, rules)

    return features, labels


def sample_data_from_indices(samples, rules, n, m, s, L, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n 	# div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor')	# div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl					# compute remainder (run in range(data_per_hl))

    labels = high_level	# labels are the classes (features of highest level)
    features = labels		# init input features as labels (rep. size 1)
    size = 1

    if return_tree_structure:
        features_dict = (
            {}
        )  # Initialize the dictionary to store the hidden variables
        features_dict[0] = copy.deepcopy(features)
        for l in range(L):

            choices = m**(size)
            data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

            high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
            high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

            features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
            features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
            features_dict[l+1] = copy.deepcopy(features)
            size *= s								# rep. size increases by s at each level

            low_level = low_level % data_per_hl				# compute remainder (run in range(data_per_hl))

        return features_dict, labels

    else:
        for l in range(L):

            choices = m**(size)
            data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

            high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
            high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

            features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
            features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
            size *= s								# rep. size increases by s at each level

            low_level = low_level % data_per_hl				# compute remainder (run in range(data_per_hl))
        features = features.reshape(samples.shape[0], -1)
        return features, labels



def sample_without_replacement(max_data, train_size, test_size, seed_sample, rules):

    L = len(rules)  # Number of levels in the hierarchy
    n = rules[0].shape[0]  # Number of classes
    m = rules[0].shape[1]  # Number of synonymic lower-level representations
    s = rules[0].shape[2]  # Size of lower-level representations

    if train_size == -1:
        samples = torch.arange(max_data)
    else:
        test_size = min(test_size, max_data - train_size)

        random.seed(seed_sample)
        samples = torch.tensor(random.sample(range(max_data), train_size + test_size))

    features, labels = sample_data_from_indices(samples, rules, n, m, s, L)


    return features, labels

