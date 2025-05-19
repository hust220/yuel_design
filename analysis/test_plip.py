import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDecomposition
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def analyze_ligand_pharmacophore(sdf_file: str) -> None:
    """
    Analyze ligand-based pharmacophore features from an SDF file.
    
    Args:
        sdf_file (str): Path to the SDF file containing the ligand
    """
    # Read the SDF file
    mol = Chem.SDMolSupplier(sdf_file)[0]
    if mol is None:
        print(f"Error: Could not read molecule from {sdf_file}")
        return
    
    # Generate 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Get basic molecular properties
    print("\n=== Basic Molecular Properties ===")
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"Number of bonds: {mol.GetNumBonds()}")
    print(f"Molecular weight: {rdMolDescriptors.CalcExactMolWt(mol):.2f}")
    print(f"Number of rotatable bonds: {rdMolDescriptors.CalcNumRotatableBonds(mol)}")
    print(f"Number of aromatic rings: {rdMolDescriptors.CalcNumAromaticRings(mol)}")
    
    # Analyze pharmacophore features
    print("\n=== Pharmacophore Features ===")
    
    # Hydrogen Bond Donors
    hbd = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetTotalNumHs() > 0:
            neighbor = atom.GetNeighbors()[0]
            if neighbor.GetAtomicNum() in [7, 8]:  # N or O
                hbd.append(neighbor.GetIdx())
    print(f"Hydrogen Bond Donors: {len(hbd)}")
    for idx in hbd:
        atom = mol.GetAtomWithIdx(idx)
        print(f"  {atom.GetSymbol()} at position {idx}")
    
    # Hydrogen Bond Acceptors
    hba = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8]:  # N or O
            hba.append(atom.GetIdx())
    print(f"\nHydrogen Bond Acceptors: {len(hba)}")
    for idx in hba:
        atom = mol.GetAtomWithIdx(idx)
        print(f"  {atom.GetSymbol()} at position {idx}")
    
    # Aromatic Rings
    aromatic_rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(ring)
    print(f"\nAromatic Rings: {len(aromatic_rings)}")
    for i, ring in enumerate(aromatic_rings):
        print(f"  Ring {i+1}: Atoms {ring}")
    
    # Hydrophobic Centers
    hydrophobic = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():  # Non-aromatic carbon
            hydrophobic.append(atom.GetIdx())
    print(f"\nHydrophobic Centers: {len(hydrophobic)}")
    for idx in hydrophobic:
        atom = mol.GetAtomWithIdx(idx)
        print(f"  {atom.GetSymbol()} at position {idx}")
    
    # Positive Ionizable Centers
    positive = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0:  # Positively charged nitrogen
            positive.append(atom.GetIdx())
    print(f"\nPositive Ionizable Centers: {len(positive)}")
    for idx in positive:
        atom = mol.GetAtomWithIdx(idx)
        print(f"  {atom.GetSymbol()} at position {idx}")
    
    # Negative Ionizable Centers
    negative = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8] and atom.GetFormalCharge() < 0:  # Negatively charged N or O
            negative.append(atom.GetIdx())
    print(f"\nNegative Ionizable Centers: {len(negative)}")
    for idx in negative:
        atom = mol.GetAtomWithIdx(idx)
        print(f"  {atom.GetSymbol()} at position {idx}")
    
    # Generate 2D visualization with pharmacophore features highlighted
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.DrawMolecule(mol, highlightAtoms=hbd + hba + hydrophobic + positive + negative)
    drawer.FinishDrawing()
    
    # Save the visualization
    with open('pharmacophore_features.png', 'wb') as f:
        f.write(drawer.GetDrawingText())


def test_pharmacophore_analysis():
    """Test pharmacophore analysis on a sample ligand."""
    # Example SDF file path - replace with your test file
    test_sdf = "best_generation_structures/molecule_24979_medusa-127.11_random-26.14/native_pose.sdf"
    
    if not os.path.exists(test_sdf):
        print(f"Test file {test_sdf} not found. Please provide a valid SDF file.")
        return
    
    analyze_ligand_pharmacophore(test_sdf)


if __name__ == "__main__":
    test_pharmacophore_analysis()