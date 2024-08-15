import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from ..protein_ligand import AA_NAME_SYM

AA_INDEX_TO_SYM = {i + 1: sym for i, (_, sym) in enumerate(AA_NAME_SYM.items())}
AA_INDEX_TO_SYM[0] = "X"  # Add a placeholder for index 0, if needed


def from_protein_ligand_dicts(
    large_pocket_dict=None,
    ligand_dict=None,
    residue_dict=None,
    seq=None,
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

    if seq is not None:
        instance["seq"] = seq

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
        self.db = None
        self.keys = None

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
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
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data["id"] = idx

        # Process residue data
        residue_dict = self.process_residue_data(data)
        data.update(residue_dict)
        # Ensure 'seq' is in the data
        if "seq" not in data:
            if "amino_acid" in data:
                data["seq"] = "".join(
                    [AA_INDEX_TO_SYM[aa.item()] for aa in data["amino_acid"]]
                )
            else:
                raise KeyError("Unable to determine sequence information")
        # Ensure 'protein_pos' is in the data
        assert data["protein_pos"].size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data

    def process_residue_data(self, data):
        residue_dict = {
            "amino_acid": [],
            "res_idx": [],
            "residue_natoms": [],
            "pos_N": [],
            "pos_CA": [],
            "pos_C": [],
            "pos_O": [],
        }

        current_aa_type = None
        atom_count = 0
        res_idx = 0

        for name, aa_type, pos in zip(
            data["protein_atom_name"],
            data["protein_atom_to_aa_type"],
            data["protein_pos"],
        ):
            if name in ("N", "CA", "C") and aa_type != current_aa_type:
                if current_aa_type is not None:
                    residue_dict["amino_acid"].append(current_aa_type)
                    residue_dict["res_idx"].append(res_idx)
                    residue_dict["residue_natoms"].append(atom_count)
                    res_idx += 1

                current_aa_type = aa_type.item()
                atom_count = 1
            else:
                atom_count += 1

            if name in ["N", "CA", "C", "O"]:
                residue_dict[f"pos_{name}"].append(pos)

        # Add the last residue
        if current_aa_type is not None:
            residue_dict["amino_acid"].append(current_aa_type)
            residue_dict["res_idx"].append(res_idx)
            residue_dict["residue_natoms"].append(atom_count)

        # Convert lists to tensors
        for key in residue_dict:
            if residue_dict[key]:
                residue_dict[key] = (
                    torch.stack(residue_dict[key])
                    if key.startswith("pos_")
                    else torch.tensor(residue_dict[key])
                )
            else:
                residue_dict[key] = torch.tensor([])

        residue_dict["small_pocket_residue_mask"] = torch.ones_like(
            residue_dict["amino_acid"], dtype=torch.bool
        )

        return residue_dict

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
