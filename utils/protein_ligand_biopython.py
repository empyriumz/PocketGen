import numpy as np
from Bio import PDB
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

ATOM_FAMILIES = [
    "Acceptor",
    "Donor",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
]
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}


def parse_sdf_file(path, feat=True):
    try:
        # Read the SDF file
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        mol = next(suppl)

        if mol is None:
            raise ValueError(f"Unable to read molecule from {path}")

        # Attempt to sanitize the molecule
        try:
            Chem.SanitizeMol(mol)
        except:
            print(
                f"Warning: Unable to sanitize molecule from {path}. Proceeding with unsanitized molecule."
            )

        # Get the number of atoms
        num_atoms = mol.GetNumAtoms()

        # Calculate atom features
        feat_mat = np.zeros((num_atoms, len(ATOM_FAMILIES)), dtype=np.int64)
        if feat:
            fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            features = factory.GetFeaturesForMol(mol)
            for feature in features:
                family = feature.GetFamily()
                if family in ATOM_FAMILIES_ID:
                    atom_ids = feature.GetAtomIds()
                    feat_mat[atom_ids, ATOM_FAMILIES_ID[family]] = 1

        # Get atom positions, elements, and calculate center of mass
        ptable = Chem.GetPeriodicTable()
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atomic_weights = [ptable.GetAtomicWeight(atomic_num) for atomic_num in elements]
        center_of_mass = np.average(positions, axis=0, weights=atomic_weights)
        # Get bond information
        bonds = mol.GetBonds()
        edge_index = []
        edge_type = []
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = int(bond.GetBondType())
            edge_index.extend([[i, j], [j, i]])
            edge_type.extend([bond_type, bond_type])

        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_type = np.array(edge_type, dtype=np.int64)

        # Sort edges
        perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]

        # Get neighbors
        neighbor_dict = {
            i: [n.GetIdx() for n in atom.GetNeighbors()]
            for i, atom in enumerate(mol.GetAtoms())
        }

        data = {
            "element": np.array(elements, dtype=np.int32),
            "pos": np.array(positions, dtype=np.float32),
            "bond_index": edge_index,
            "bond_type": edge_type,
            "center_of_mass": center_of_mass.astype(np.float32),
            "atom_feature": feat_mat,
            "neighbors": neighbor_dict,
        }
        return data

    except Exception as e:
        print(f"Error processing SDF file {path}: {str(e)}")
        return None


class PDBProtein:
    def __init__(self, pdb_file, remove_water=True):
        self.pdb_file = pdb_file
        self.remove_water = remove_water
        self.structure = None
        self.residues = []
        self.atoms = []
        self.seq = []
        self.amino_acid = []
        self.amino_idx = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []
        self.residue_natoms = []
        self.dssp = {}

        self._parse()

    def _parse(self):
        parser = PDB.PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", self.pdb_file)

        # Remove water if specified
        if self.remove_water:
            for model in self.structure:
                for chain in model:
                    residues_to_remove = [r for r in chain if r.get_resname() == "HOH"]
                    for r in residues_to_remove:
                        chain.detach_child(r.id)

        # Parse residues and atoms
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue, standard=True):
                        res_dict = {
                            "name": residue.get_resname(),
                            "id": residue.get_id()[1],
                            "atoms": [],
                        }

                        for atom in residue:
                            if atom.name not in ["H", "OXT"]:
                                atom_dict = {
                                    "name": atom.name,
                                    "element": atom.element,
                                    "pos": atom.coord,
                                }
                                res_dict["atoms"].append(atom_dict)
                                self.atoms.append(atom_dict)

                        self.residues.append(res_dict)
                        self.seq.append(
                            PDB.Polypeptide.protein_letters_3to1[residue.get_resname()]
                        )
                        self.amino_acid.append(
                            PDB.Polypeptide.three_to_index(residue.get_resname())
                        )
                        self.amino_idx.append(residue.get_id()[1])
                        self.residue_natoms.append(len(res_dict["atoms"]))

                        # Get backbone atom positions
                        self.pos_CA.append(
                            residue["CA"].coord if "CA" in residue else None
                        )
                        self.pos_C.append(
                            residue["C"].coord if "C" in residue else None
                        )
                        self.pos_N.append(
                            residue["N"].coord if "N" in residue else None
                        )
                        self.pos_O.append(
                            residue["O"].coord if "O" in residue else None
                        )

                        # Calculate center of mass
                        total_mass = sum(a.mass for a in residue)
                        weighted_coords = sum(a.mass * a.coord for a in residue)
                        self.center_of_mass.append(weighted_coords / total_mass)

    def to_dict_atom(self):
        return {
            "element": np.array(
                [PDB.atom.atomic_number[a["element"]] for a in self.atoms],
                dtype=np.int64,
            ),
            "pos": np.array([a["pos"] for a in self.atoms], dtype=np.float32),
            "atom_name": np.array([a["name"] for a in self.atoms]),
            "atom_to_aa_type": np.array(
                [
                    PDB.Polypeptide.three_to_index(r["name"])
                    for r in self.residues
                    for _ in r["atoms"]
                ],
                dtype=np.int64,
            ),
        }

    def to_dict_residue(self):
        return {
            "seq": self.seq,
            "res_idx": np.array(self.amino_idx, dtype=np.int64),
            "amino_acid": np.array(self.amino_acid, dtype=np.int64),
            "center_of_mass": np.array(self.center_of_mass, dtype=np.float32),
            "pos_CA": np.array(
                [p for p in self.pos_CA if p is not None], dtype=np.float32
            ),
            "pos_C": np.array(
                [p for p in self.pos_C if p is not None], dtype=np.float32
            ),
            "pos_N": np.array(
                [p for p in self.pos_N if p is not None], dtype=np.float32
            ),
            "pos_O": np.array(
                [p for p in self.pos_O if p is not None], dtype=np.float32
            ),
            "residue_natoms": np.array(self.residue_natoms, dtype=np.int64),
        }

    def query_residues_radius(self, center, radius, criterion="center_of_mass"):
        center = np.array(center).reshape(3)
        selected = []
        for i, residue in enumerate(self.residues):
            if criterion == "center_of_mass":
                distance = np.linalg.norm(self.center_of_mass[i] - center)
            else:
                distance = min(
                    np.linalg.norm(atom["pos"] - center) for atom in residue["atoms"]
                )
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(
        self, ligand, radius=3.5, selected_residue=None, return_mask=True
    ):
        if selected_residue is None:
            selected_residue = self.residues

        selected = []
        sel_idx = set()
        selected_mask = np.zeros(len(self.residues), dtype=bool)

        for i, residue in enumerate(selected_residue):
            for ligand_pos in ligand["pos"]:
                distance = min(
                    np.linalg.norm(atom["pos"] - ligand_pos)
                    for atom in residue["atoms"]
                )
                if distance <= radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
                    break

        selected_mask[list(sel_idx)] = True

        if return_mask:
            return list(sel_idx), selected_mask
        return list(sel_idx), selected

    def get_selected_structure(self, selected_residues):
        new_structure = PDB.Structure.Structure("pocket")
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)

        for chain in self.structure[0]:
            new_chain = PDB.Chain.Chain(chain.id)
            for residue in chain:
                if any(residue.id[1] == r["id"] for r in selected_residues):
                    new_chain.add(residue.copy())
            if len(new_chain):
                new_model.add(new_chain)

        return new_structure
