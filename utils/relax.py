#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

os.environ["OPENMM_CPU_THREADS"] = "4"  # prevent openmm from using all cpus available
from openmm import LangevinIntegrator, Platform, CustomExternalForce
from openmm.app import PDBFile, Simulation, ForceField, HBonds, Modeller
from openmm.unit import kilocalories_per_mole, angstroms, picosecond, kelvin
from pdbfixer import PDBFixer

import logging

logging.getLogger("openmm").setLevel(logging.ERROR)

FILE_DIR = os.path.abspath(os.path.split(__file__)[0])

CACHE_DIR = ".data/saved/"


def openmm_relax(pdb, out_pdb=None, excluded_chains=None, inverse_exclude=False):
    tolerance = 2.39 * kilocalories_per_mole / angstroms
    stiffness = 10.0 * kilocalories_per_mole / (angstroms**2)
    if excluded_chains is None:
        excluded_chains = []
    if out_pdb is None:
        out_pdb = pdb

    fixer = PDBFixer(pdb)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    force_field = ForceField("amber99sb.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology, constraints=HBonds)

    force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    # add flexible atoms
    for residue in modeller.topology.residues():
        if (not inverse_exclude and residue.chain.id in excluded_chains) or (
            inverse_exclude and residue.chain.id not in excluded_chains
        ):  # antigen
            for atom in residue.atoms():
                system.setParticleMass(atom.index, 0)

        for atom in residue.atoms():
            if atom.element.name != "hydrogen":
                force.addParticle(atom.index, modeller.positions[atom.index])

    system.addForce(force)
    integrator = LangevinIntegrator(0 * kelvin, 0.01 / picosecond, 0.0 * picosecond)

    # Set up CUDA platform
    try:
        platform = Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "mixed", "CudaDeviceIndex": "0"}
    except Exception:
        print("CUDA platform not available. Falling back to CPU.")
        platform = Platform.getPlatformByName("CPU")
        properties = {}

    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    with open(out_pdb, "w") as fout:
        PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)

    return out_pdb


def relax_sdf(sdf):
    mol = Chem.MolFromMolFile(sdf, sanitize=True)
    mol = Chem.AddHs(mol, addCoords=True)
    UFFOptimizeMolecule(mol)

    # Save to MOL file
    Chem.MolToMolFile(mol, sdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, default="./saved/3upr_C_rec.pdb")
    parser.add_argument(
        "--sdf", type=str, default="./saved/3upr_C_rec_3vri_1kx_lig_tt_min_0.sdf"
    )
    args = parser.parse_args()

    openmm_relax(args.pdb, out_pdb=args.pdb)
    relax_sdf(args.sdf, out_sdf=args.sdf)
