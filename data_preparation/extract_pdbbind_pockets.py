import os
import argparse
import multiprocessing as mp
import pickle
import shutil
import traceback
from Bio import PDB
import pandas as pd
from functools import partial
from tqdm.auto import tqdm
from utils.protein_ligand import PDBProtein, parse_sdf_file


def process_item(item, args):
    pdb_id, category = item["pdb_id"], item["category"]
    try:
        pdb_file = os.path.join(args.source, category, pdb_id, f"{pdb_id}_protein.pdb")
        protein = PDBProtein(pdb_file, remove_water=True)
        seq = "".join(protein.to_dict_residue()["seq"])
        ligand_file = f"{pdb_id}_ligand.sdf"
        ligand_path = os.path.join(args.source, category, pdb_id, ligand_file)
        ligand = parse_sdf_file(ligand_path)

        large_pocket_idx, r10_residues = protein.query_residues_ligand(
            ligand, args.large_radius, return_mask=False
        )
        assert len(large_pocket_idx) == len(r10_residues)

        small_pocket_idx, _ = protein.query_residues_ligand(
            ligand,
            radius=args.small_radius,
            return_mask=False,
        )
        pocket_structure = protein.get_selected_structure(r10_residues)
        large_pocket_file = f"{pdb_id}_pocket{args.large_radius}.pdb"
        dest_folder = os.path.join(args.dest, "test", pdb_id)
        os.makedirs(dest_folder, exist_ok=True)
        ligand_dest = os.path.join(dest_folder, ligand_file)
        large_pocket_dest = os.path.join(dest_folder, large_pocket_file)

        # Copy ligand file
        shutil.copyfile(ligand_path, ligand_dest)
        # Copy protein file
        shutil.copyfile(pdb_file, os.path.join(dest_folder, f"{pdb_id}_protein.pdb"))
        # Save pocket structure
        io = PDB.PDBIO()
        io.set_structure(pocket_structure)
        io.save(large_pocket_dest)

        return True, (
            large_pocket_file,
            ligand_file,
            f"{pdb_id}_protein.pdb",
            seq,
            small_pocket_idx,
            large_pocket_idx,
        )
    except Exception as e:
        print(f"Exception occurred for {pdb_id}: {str(e)}")
        print(traceback.format_exc())
        return False, pdb_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the root directory containing category folders",
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Specify if train or val or test split",
    ),
    parser.add_argument(
        "--dataframe",
        type=str,
        required=True,
        help="Path to the CSV file containing the DataFrame",
    )
    parser.add_argument("--large_radius", type=int, default=10)
    parser.add_argument("--small_radius", type=float, default=3.5)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    # Load the DataFrame
    df = pd.read_csv(args.dataframe)

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    failed_ids = []

    for success, result in tqdm(
        pool.imap_unordered(partial(process_item, args=args), df.to_dict("records")),
        total=len(df),
        desc="Processing items",
    ):
        if success:
            index_pocket.append(result)
        else:
            failed_ids.append(result)  # result is pdb_id in this case

    pool.close()
    pool.join()

    # Write successful results
    index_path = os.path.join(args.dest, args.split, "index_seq.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index_pocket, f)

    # Write failed IDs to a text file
    failed_ids_path = os.path.join(args.dest, "failed_ids.txt")
    with open(failed_ids_path, "w") as f:
        for pdb_id in failed_ids:
            f.write(f"{pdb_id}\n")

    print(f"Done. {len(index_pocket)} protein-ligand pairs processed successfully.")
    print(
        f"{len(failed_ids)} protein-ligand pairs failed. See {failed_ids_path} for details."
    )
