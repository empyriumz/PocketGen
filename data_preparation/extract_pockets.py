import os
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial
from tqdm.auto import tqdm
from utils.protein_ligand import PDBProtein, parse_sdf_file


def load_pdb_block(root_path, pdb_file):
    pdb_path = os.path.join(root_path, pdb_file)
    with open(pdb_path, "r") as f:
        pdb_block = f.read()
    return pdb_block


def process_item(item, args):
    pdb_file, ligand_file = item[0], item[1]
    try:
        pdb_block = load_pdb_block(args.source, pdb_file)
        protein = PDBProtein(pdb_block)
        seq = "".join(protein.to_dict_residue()["seq"])
        ligand = parse_sdf_file(os.path.join(args.source, ligand_file))

        large_pocket_idx, r10_residues = protein.query_residues_ligand(
            ligand, args.large_radius, selected_residue=None, return_mask=False
        )
        assert len(large_pocket_idx) == len(r10_residues)

        pdb_block_pocket = protein.residues_to_pdb_block(r10_residues)

        small_pocket_idx, _ = protein.query_residues_ligand(
            ligand,
            radius=args.small_radius,
            selected_residue=r10_residues,
            return_mask=False,
        )

        pocket_file = ligand_file[:-4] + f"_pocket{args.large_radius}.pdb"
        ligand_dest = os.path.join(args.dest, ligand_file)
        pocket_dest = os.path.join(args.dest, pocket_file)

        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)
        shutil.copyfile(
            src=os.path.join(args.source, ligand_file),
            dst=ligand_dest,
        )

        with open(pocket_dest, "w") as f:
            f.write(pdb_block_pocket)

        return (
            pocket_file,
            ligand_file,
            pdb_file,
            seq,
            small_pocket_idx,
            large_pocket_idx,
        )

    except Exception as e:
        print(f"Exception occurred for {pdb_file}: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/crossdocked_v1.1_rmsd1.0")
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        default="data/crossdocked_v1.1_rmsd1.0_pocket10",
    )
    parser.add_argument("--large_radius", type=int, default=10)
    parser.add_argument("--small_radius", type=float, default=3.5)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    with open(os.path.join(args.source, "index.pkl"), "rb") as f:
        index = pickle.load(f)

    pool = mp.Pool(args.num_workers)
    index_pocket = []

    for item_pocket in tqdm(
        pool.imap_unordered(partial(process_item, args=args), index),
        total=len(index),
        desc="Processing items",
    ):
        if item_pocket is not None:
            index_pocket.append(item_pocket)

    pool.close()
    pool.join()

    index_path = os.path.join(args.dest, "index_seq.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index_pocket, f)

    print(f"Done. {len(index_pocket)} protein-ligand pairs processed successfully.")
