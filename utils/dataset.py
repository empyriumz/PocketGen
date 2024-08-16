import lmdb
import pickle
import os
from torch.utils.data import Dataset


def from_protein_ligand_dicts(
    large_pocket_dict=None,
    ligand_dict=None,
    residue_dict=None,
    full_seq=None,
    small_pocket_idx=None,
    large_pocket_idx=None,
):
    instance = {}

    if large_pocket_dict is not None:
        for key, item in large_pocket_dict.items():
            instance["protein_" + key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            instance["ligand_" + key] = item

    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item

    if full_seq is not None:
        instance["full_seq"] = full_seq

    if small_pocket_idx is not None:
        instance["small_pocket_idx"] = small_pocket_idx

    if large_pocket_idx is not None:
        instance["large_pocket_idx"] = large_pocket_idx

    return instance


class PocketLigandPairDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.keys = None

        self._connect_db()

    def _connect_db(self):
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"The LMDB file '{self.lmdb_path}' does not exist.")

        try:
            self.env = lmdb.open(
                self.lmdb_path,
                max_readers=1,
                readonly=True,
                subdir=False,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.env.begin(write=False) as txn:
                self.keys = list(txn.cursor().iternext(values=False))
        except lmdb.Error as e:
            raise lmdb.Error(f"Error opening LMDB file: {e}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(key))

        data["id"] = idx
        if self.transform:
            data = self.transform(data)
        return data

    def __del__(self):
        if self.env:
            self.env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the LMDB file")
    args = parser.parse_args()
    try:
        dataset = PocketLigandPairDataset(args.path)
        print(f"Dataset size: {len(dataset)}")
        print(f"First item keys: {dataset[0].keys()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the file path is correct and the file exists.")
    except lmdb.Error as e:
        print(f"LMDB Error: {e}")
        print("Please ensure the file is a valid LMDB database.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
