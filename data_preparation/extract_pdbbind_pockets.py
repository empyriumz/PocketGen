import os
import torch
import numpy as np
import argparse
import multiprocessing as mp
import lmdb
import pickle
from rdkit import Chem
import pandas as pd
from Bio import PDB
from functools import partial
from tqdm.auto import tqdm
from utils.protein_ligand import PDBProtein, parse_sdf_file


def validate_data(data):
    try:
        assert data["protein_pos"].shape[0] > 0, "No protein atoms"
        assert len(data["large_pocket_idx"]) == len(
            data["res_idx"]
        ), f"Mismatch in residue counts: large_pocket_idx ({len(data['large_pocket_idx'])}) != res_idx ({len(data['res_idx'])})"
        assert (
            data["pos_N"].shape[0]
            == data["pos_CA"].shape[0]
            == data["pos_C"].shape[0]
            == data["pos_O"].shape[0]
        ), "Mismatch in backbone coordinate counts"
        assert data["small_pocket_residue_mask"].sum() == len(
            data["small_pocket_global_idx"]
        ), "Mismatch in small pocket residue counts"
        assert (
            data["ligand_pos"].shape[0] == data["ligand_element"].shape[0]
        ), "Mismatch in ligand atom counts"
        assert (
            data["ligand_bond_index"].shape[1] == data["ligand_bond_type"].shape[0]
        ), "Mismatch in bond data"
        assert set(data["small_pocket_global_idx"]).issubset(
            set(data["large_pocket_idx"])
        ), "Small pocket is not a subset of large pocket"
        return True
    except AssertionError as e:
        print(f"Validation failed: {str(e)} for {data['protein_filename']}")
        return False


def process_item(item, args):
    pdb_id, category = item["pdb_id"], item["category"]
    try:
        pdb_file = os.path.join(args.source, category, pdb_id, f"{pdb_id}_protein.pdb")
        protein = PDBProtein(pdb_file, remove_water=True)
        full_seq = "".join(protein.seq)
        ligand_file = f"{pdb_id}_ligand.sdf"
        ligand_path = os.path.join(args.source, category, pdb_id, ligand_file)
        ligand_dict = parse_sdf_file(ligand_path)
        if ligand_dict is None:
            return False, pdb_id

        # Get pocket residues
        large_pocket_idx, large_pocket_residues = protein.query_residues_ligand(
            ligand_dict, args.large_radius, return_mask=False
        )
        small_pocket_global_idx, _ = protein.query_residues_ligand(
            ligand_dict, radius=args.small_radius, return_mask=True
        )
        small_pocket_relative_idx = [
            large_pocket_idx.index(idx)
            for idx in small_pocket_global_idx
            if idx in large_pocket_idx
        ]
        small_pocket_relative_mask = np.zeros(len(large_pocket_idx), dtype=bool)
        small_pocket_relative_mask[small_pocket_relative_idx] = True

        pocket_atom_dict = {
            "element": [],
            "pos": [],
            "atom_name": [],
            "atom_to_aa_type": [],
        }
        pocket_residue_dict = {
            "seq": [],
            "res_idx": [],
            "amino_acid": [],
            "center_of_mass": [],
            "pos_CA": [],
            "pos_C": [],
            "pos_N": [],
            "pos_O": [],
            "residue_num_atoms": [],
        }

        ptable = Chem.GetPeriodicTable()

        for residue in large_pocket_residues:
            pocket_residue_dict["seq"].append(
                PDB.Polypeptide.protein_letters_3to1[residue["name"]]
            )
            pocket_residue_dict["res_idx"].append(residue["id"])
            pocket_residue_dict["amino_acid"].append(
                PDB.Polypeptide.three_to_index(residue["name"])
            )

            atom_positions = []
            atom_count = 0
            for atom in residue["atoms"]:
                pocket_atom_dict["element"].append(
                    ptable.GetAtomicNumber(atom["element"])
                )
                pocket_atom_dict["pos"].append(atom["pos"])
                pocket_atom_dict["atom_name"].append(atom["name"])
                pocket_atom_dict["atom_to_aa_type"].append(
                    PDB.Polypeptide.three_to_index(residue["name"])
                )
                atom_positions.append(atom["pos"])
                atom_count += 1

            pocket_residue_dict["residue_num_atoms"].append(atom_count)
            pocket_residue_dict["center_of_mass"].append(
                np.mean(atom_positions, axis=0)
            )

            pocket_residue_dict["pos_CA"].append(
                next((a["pos"] for a in residue["atoms"] if a["name"] == "CA"), None)
            )
            pocket_residue_dict["pos_C"].append(
                next((a["pos"] for a in residue["atoms"] if a["name"] == "C"), None)
            )
            pocket_residue_dict["pos_N"].append(
                next((a["pos"] for a in residue["atoms"] if a["name"] == "N"), None)
            )
            pocket_residue_dict["pos_O"].append(
                next((a["pos"] for a in residue["atoms"] if a["name"] == "O"), None)
            )

        # Convert lists to numpy arrays
        for key in pocket_atom_dict:
            pocket_atom_dict[key] = np.array(pocket_atom_dict[key])
        for key in pocket_residue_dict:
            if key in ["pos_N", "pos_CA", "pos_C", "pos_O"]:
                # Filter out None values
                pocket_residue_dict[key] = [
                    p for p in pocket_residue_dict[key] if p is not None
                ]
            else:
                pocket_residue_dict[key] = np.array(pocket_residue_dict[key])

        # Prepare data for LMDB
        data = {
            "protein_pos": torch.tensor(pocket_atom_dict["pos"], dtype=torch.float32),
            "protein_atom_feature": torch.tensor(
                pocket_atom_dict["element"], dtype=torch.long
            ),
            "amino_acid": torch.tensor(
                pocket_residue_dict["amino_acid"], dtype=torch.long
            ),
            "residue_num_atoms": torch.tensor(
                pocket_residue_dict["residue_num_atoms"], dtype=torch.long
            ),
            "protein_atom_to_aa_type": torch.tensor(
                pocket_atom_dict["atom_to_aa_type"], dtype=torch.long
            ),
            "res_idx": torch.tensor(pocket_residue_dict["res_idx"], dtype=torch.long),
            "ligand_element": torch.tensor(ligand_dict["element"], dtype=torch.long),
            "ligand_pos": torch.tensor(ligand_dict["pos"], dtype=torch.float32),
            "ligand_atom_feature": torch.tensor(
                ligand_dict["atom_feature"], dtype=torch.float32
            ),
            "ligand_bond_index": torch.tensor(
                ligand_dict["bond_index"], dtype=torch.long
            ),
            "ligand_bond_type": torch.tensor(
                ligand_dict["bond_type"], dtype=torch.long
            ),
            "pos_N": torch.tensor(
                pocket_residue_dict["pos_N"],
                dtype=torch.float32,
            ),
            "pos_CA": torch.tensor(
                pocket_residue_dict["pos_CA"],
                dtype=torch.float32,
            ),
            "pos_C": torch.tensor(
                pocket_residue_dict["pos_C"],
                dtype=torch.float32,
            ),
            "pos_O": torch.tensor(
                pocket_residue_dict["pos_O"],
                dtype=torch.float32,
            ),
            "small_pocket_residue_mask": torch.tensor(
                small_pocket_relative_mask, dtype=torch.bool
            ),
            "small_pocket_global_idx": small_pocket_global_idx,
            "small_pocket_relative_idx": torch.tensor(
                small_pocket_relative_idx, dtype=torch.long
            ),
            "large_pocket_idx": large_pocket_idx,
            "protein_atom_name": pocket_atom_dict["atom_name"],
            "full_seq": full_seq,
            "ligand_filename": ligand_file,
            "protein_filename": f"{pdb_id}_protein.pdb",
        }

        # Validate data
        is_valid = validate_data(data)
        # convert to tensor later to avoid assertion error
        data["small_pocket_global_idx"] = torch.tensor(
            data["small_pocket_global_idx"], dtype=torch.long
        )
        data["large_pocket_idx"] = torch.tensor(
            data["large_pocket_idx"], dtype=torch.long
        )
        if not is_valid:
            return False, pdb_id

        return True, (pdb_id, pickle.dumps(data))
    except Exception as e:
        print(f"Exception occurred for {pdb_id}: {str(e)}")
        return False, pdb_id


def main(args):
    df = pd.read_csv(args.dataframe)
    lmdb_path = f"{args.dest}_{args.split}.lmdb"
    # Create LMDB environment
    env = lmdb.open(
        lmdb_path,
        map_size=10 * (1024 * 1024 * 1024),  # 10GB
        create=True,
        subdir=False,
        readonly=False,
    )

    pool = mp.Pool(args.num_workers)
    failed_ids = []

    with env.begin(write=True) as txn:
        for success, result in tqdm(
            pool.imap_unordered(
                partial(process_item, args=args), df.to_dict("records")
            ),
            total=len(df),
            desc="Processing items",
        ):
            if success:
                pdb_id, data = result
                txn.put(pdb_id.encode(), data)
            else:
                failed_ids.append(result)

    pool.close()
    pool.join()

    # Write failed IDs to a text file
    failed_ids_path = os.path.join(
        os.path.dirname(args.dest), f"failed_ids_{args.split}.txt"
    )
    with open(failed_ids_path, "w") as f:
        for pdb_id in failed_ids:
            f.write(f"{pdb_id}\n")

    print(
        f"Done. {len(df) - len(failed_ids)} protein-ligand pairs processed successfully."
    )
    print(
        f"{len(failed_ids)} protein-ligand pairs failed. See {failed_ids_path} for details."
    )
    print(f"LMDB file created at: {lmdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the root directory containing category folders",
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Path to the output LMDB file"
    )
    parser.add_argument(
        "--dataframe",
        type=str,
        required=True,
        help="Path to the CSV file containing the DataFrame",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Specify if train or val or test split",
    ),
    parser.add_argument("--large_radius", type=int, default=10)
    parser.add_argument("--small_radius", type=float, default=3.5)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    main(args)
