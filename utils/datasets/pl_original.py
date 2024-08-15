import os
import pickle
import lmdb
import traceback
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from ..protein_ligand import PDBProtein, parse_sdf_file
from ..data import torchify_dict


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

    def __init__(self, raw_path, pocket_size=3.5, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip("/")
        self.pocket_size = pocket_size
        self.index_path = os.path.join(self.raw_path, "index_seq.pkl")
        self.processed_path = os.path.join(
            os.path.dirname(self.raw_path),
            os.path.basename(self.raw_path) + "_processed.lmdb",
        )
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process()

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
        self.db = lmdb.open(
            self.processed_path,
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

    def _validate_data(
        self,
        large_pocket_dict,
        residue_dict,
        ligand_dict,
        small_pocket_idx,
        large_pocket_idx,
        small_pocket_residue_mask,
    ):
        try:
            # Check protein data
            backbone_keys = ["pos_N", "pos_CA", "pos_C", "pos_O"]
            backbone_sizes = [
                residue_dict[key].shape[0]
                for key in backbone_keys
                if key in residue_dict
            ]
            assert (
                len(backbone_sizes) == 4 and len(set(backbone_sizes)) == 1
            ), "Inconsistent backbone sizes"

            # Check small_pocket_idx is a subset of large_pocket_idx
            assert set(small_pocket_idx).issubset(
                set(large_pocket_idx)
            ), "small_pocket_idx is not a subset of large_pocket_idx"

            # Check atom counts
            assert large_pocket_dict["element"].shape[0] == sum(
                residue_dict["residue_natoms"]
            ), "Mismatch in atom counts"

            # Check ligand data
            required_ligand_keys = [
                "element",
                "pos",
                "bond_index",
                "bond_type",
                "center_of_mass",
                "atom_feature",
            ]
            assert all(
                key in ligand_dict for key in required_ligand_keys
            ), "Missing required ligand keys"

            num_ligand_atoms = ligand_dict["element"].shape[0]
            assert (
                ligand_dict["pos"].shape[0] == num_ligand_atoms
            ), "Mismatch in ligand atom counts"
            assert (
                ligand_dict["atom_feature"].shape[0] == num_ligand_atoms
            ), "Mismatch in ligand atom feature counts"

            # Check bond data
            assert (
                ligand_dict["bond_index"].shape[1] == ligand_dict["bond_type"].shape[0]
            ), "Mismatch in bond data"
            assert (
                ligand_dict["bond_index"].max() < num_ligand_atoms
            ), "Invalid bond index"

            # Check center_of_mass
            assert ligand_dict["center_of_mass"].shape == (
                3,
            ), "Invalid center_of_mass shape"

            assert small_pocket_residue_mask.sum() > 0, "No residues selected"
            assert small_pocket_residue_mask.sum() == len(
                small_pocket_idx
            ), "Mismatch in selected residues"

            return True

        except AssertionError as e:
            print(f"Validation failed: {str(e)}")
            return False

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, "rb") as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (
                pocket_fn,
                ligand_fn,
                protein_fn,
                full_seq,
                small_pocket_idx,
                large_pocket_idx,
            ) in enumerate(tqdm(index)):
                if pocket_fn is None:
                    continue
                try:
                    pdb_id = pocket_fn.split("_")[0]
                    large_pocket = PDBProtein(
                        os.path.join(self.raw_path, pdb_id, pocket_fn)
                    )
                    large_pocket_dict = large_pocket.to_dict_atom()
                    residue_dict = large_pocket.to_dict_residue()
                    ligand_dict = parse_sdf_file(
                        os.path.join(self.raw_path, pdb_id, ligand_fn)
                    )
                    _, small_pocket_residue_mask = large_pocket.query_residues_ligand(
                        ligand_dict, radius=self.pocket_size, return_mask=True
                    )
                    # Validate data
                    is_valid = self._validate_data(
                        large_pocket_dict,
                        residue_dict,
                        ligand_dict,
                        small_pocket_idx,
                        large_pocket_idx,
                        small_pocket_residue_mask,
                    )

                    if not is_valid:
                        num_skipped += 1
                        print(
                            f"Skipping {pdb_id} due to validation failure {num_skipped} / {i}"
                        )
                        continue

                    residue_dict["small_pocket_residue_mask"] = (
                        small_pocket_residue_mask
                    )

                    data = from_protein_ligand_dicts(
                        large_pocket_dict=torchify_dict(large_pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict),
                        full_seq=full_seq,
                        small_pocket_idx=torch.tensor(small_pocket_idx),
                        large_pocket_idx=torch.tensor(large_pocket_idx),
                    )
                    data["ligand_filename"] = ligand_fn
                    data["protein_filename"] = protein_fn
                    txn.put(key=str(i).encode(), value=pickle.dumps(data))

                except Exception as e:
                    print(f"Error processing {pdb_id}: {str(e)}")
                    traceback.print_exc()
                    num_skipped += 1
                    continue
        db.close()

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
        assert data["protein_pos"].size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    PocketLigandPairDataset(args.path)
