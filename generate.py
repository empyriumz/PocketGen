import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from vina import Vina
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.evaluation.docking_vina import PrepLig, PrepProt
from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
from utils.misc import load_config, seed_all
from utils.dataset import from_protein_ligand_dicts
from utils.data import torchify_dict, BatchConverter, Alphabet, collate_mols_block
from models.Pocket_Design import Pocket_Design


def setup_vina():
    return Vina(sf_name="vina", seed=0, verbosity=0)


def prepare_ligand(lig_path):
    ligand_pdbqt = lig_path.replace(".sdf", ".pdbqt")
    lig = PrepLig(lig_path, "sdf")
    lig.addH()
    lig.get_pdbqt(ligand_pdbqt)
    return ligand_pdbqt


def prepare_protein(pro_path):
    protein_pqr = pro_path.replace(".pdb", ".pqr")
    protein_pdbqt = pro_path.replace(".pdb", ".pdbqt")
    prot = PrepProt(pro_path)
    prot.addH(protein_pqr)
    prot.get_pdbqt(protein_pdbqt)
    return protein_pdbqt


def calculate_vina_score(v, protein_pdbqt, ligand_pdbqt, ligand_pos):
    v.set_receptor(protein_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)

    center = np.mean(ligand_pos, 0)
    size = (ligand_pos.max(0) - ligand_pos.min(0)) * 1.2 + 8.0
    v.compute_vina_maps(center=center, box_size=size)

    energy = v.score()
    energy_minimized = v.optimize()
    v.dock(exhaustiveness=64, n_poses=30)
    score = v.energies(n_poses=1)[0][0]

    return score


def process_protein(name, args, transform=None):
    pdb_path = os.path.join(args.target, name, f"{name}.pdb")
    lig_path = os.path.join(args.target, name, f"{name}_ligand.sdf")
    pocket_path = os.path.join(args.target, name, f"{name}_pocket.pdb")

    with open(pdb_path, "r") as f:
        pdb_block = f.read()

    protein = PDBProtein(pdb_block)
    seq = "".join(protein.to_dict_residue()["seq"])
    ligand = parse_sdf_file(lig_path, feat=False)

    large_pocket_idx, larget_pocket_residues = protein.query_residues_ligand(
        ligand, radius=10, return_mask=False
    )
    small_pocket_idx, _ = protein.query_residues_ligand(
        ligand, radius=3.5, selected_residue=larget_pocket_residues, return_mask=False
    )

    pdb_block_pocket = protein.residues_to_pdb_block(larget_pocket_residues)
    with open(pocket_path, "w") as f:
        f.write(pdb_block_pocket)

    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = pocket.to_dict_atom()
    residue_dict = pocket.to_dict_residue()
    _, residue_dict["small_pocket_residue_mask"] = pocket.query_residues_ligand(ligand)

    data = from_protein_ligand_dicts(
        large_pocket_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand),
        residue_dict=torchify_dict(residue_dict),
        seq=seq,
        small_pocket_idx=torch.tensor(sorted(small_pocket_idx)),
        large_pocket_idx=torch.tensor(sorted(large_pocket_idx)),
    )
    data.update(
        {
            "protein_filename": pocket_path,
            "ligand_filename": lig_path,
            "whole_protein_name": pdb_path,
        }
    )
    if transform is not None:
        return transform(data)
    else:
        return data


def main(args):
    config = load_config(args.config)
    seed_all(2023)

    transform = Compose([FeaturizeProteinAtom(), FeaturizeLigandAtom()])
    alphabet = Alphabet.from_architecture("ESM-1b")
    batch_converter = BatchConverter(alphabet)

    model = Pocket_Design(
        config.model,
        protein_atom_feature_dim=FeaturizeProteinAtom().feature_dim,
        ligand_atom_feature_dim=FeaturizeLigandAtom().feature_dim,
        device=args.device,
    ).to(args.device)
    model.load_state_dict(
        torch.load(config.model.checkpoint, map_location=args.device)["model"]
    )

    names = ["2p16"]
    vina = setup_vina()

    for name in tqdm(names):
        data = process_protein(name, args, transform=transform)
        datalist = [data for _ in range(8)]

        dir_name = os.path.dirname(data["protein_filename"])
        os.makedirs(dir_name, exist_ok=True)

        test_loader = DataLoader(
            datalist,
            batch_size=4,
            shuffle=False,
            num_workers=config.train.num_workers,
            collate_fn=partial(collate_mols_block, batch_converter=batch_converter),
        )

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating"):
                batch = {
                    k: v.to(args.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                model.generate(batch, dir_name)

        protein_pdbqt = prepare_protein(data["protein_filename"])
        ligand_pdbqt = prepare_ligand(data["ligand_filename"])
        ligand_pos = parse_sdf_file(data["ligand_filename"], feat=False)["pos"]

        score = calculate_vina_score(vina, protein_pdbqt, ligand_pdbqt, ligand_pos)
        print(f"Vina score for {name}: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_model.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target", type=str, default="./examples")
    args = parser.parse_args()

    main(args)
