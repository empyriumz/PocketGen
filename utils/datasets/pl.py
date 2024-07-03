import pickle
import lmdb
from torch.utils.data import Dataset


def from_protein_ligand_dicts(
    protein_dict=None,
    ligand_dict=None,
    residue_dict=None,
    seq=None,
    full_seq_idx=None,
    r10_idx=None,
):
    instance = {}

    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance["protein_" + key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            instance["ligand_" + key] = item

    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item

    if seq is not None:
        instance["seq"] = seq

    if full_seq_idx is not None:
        instance["full_seq_idx"] = full_seq_idx

    if r10_idx is not None:
        instance["r10_idx"] = r10_idx

    return instance


class PocketLigandPairDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.db = None
        self.keys = None
        self._connect_db()

    def _connect_db(self):
        self.db = lmdb.open(
            self.lmdb_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data["id"] = idx
        assert data["protein_pos"].size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __del__(self):
        if self.db is not None:
            self._close_db()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the .lmdb file")
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(f"Dataset size: {len(dataset)}")
    print(f"First item keys: {dataset[0].keys()}")
