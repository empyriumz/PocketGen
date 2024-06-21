import torch
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset


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
    name = config["name"]
    root = config["path"]

    # Initialize dataset based on name
    if name == "pl":
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    # Handle dataset splitting
    subsets = {}
    if "split" in config:
        split = torch.load(config["split"])
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}

    return dataset, subsets
