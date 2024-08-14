import torch
from torch.utils.data import Subset
from .pl_original import PocketLigandPairDataset


def get_dataset(config, *args, **kwargs):
    """
    Fetches the dataset based on the configuration.

    Parameters:
    - config: A configuration object or dictionary with at least 'name' and 'path' keys,
              and optionally 'split' to specify how to split the dataset.

    Returns:
    - A tuple (dataset, subsets) where:
        - dataset is an instance of the dataset requested.
        - subsets is a dictionary of subsets (e.g., train, validation) if splits are provided,
          otherwise an empty dictionary.
    """
    train_dataset = PocketLigandPairDataset(config["train_path"], *args, **kwargs)
    val_dataset = PocketLigandPairDataset(config["val_path"], *args, **kwargs)

    return train_dataset, val_dataset
