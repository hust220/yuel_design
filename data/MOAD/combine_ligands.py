import os
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

def disable_rdkit_logging():
    """Disable RDKit logging to avoid unnecessary output."""
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

def run(ligands_dir, output_conformers):
    """Process ligands and write conformers to an output file."""
    conformers = []
    # Use tqdm to show progress while iterating over files in the directory
    for fname in tqdm(os.listdir(ligands_dir), desc="Processing ligands"):
        if fname.endswith('.mol') and not fname.startswith('._'):
            mol_name = fname.split('.')[0]
            try:
                # Read and sanitize the molecule
                mol = Chem.MolFromMolFile(os.path.join(ligands_dir, fname))
                mol = Chem.RemoveAllHs(mol)
                Chem.SanitizeMol(mol)
            except Exception as e:
                # Skip invalid molecules
                continue
            if mol is None:
                continue

            # Filter molecules based on criteria
            if mol.GetNumAtoms() <= 40 and mol.GetRingInfo().NumRings() >= 2:
                mol.SetProp('_Name', mol_name)
                conformers.append(mol)

    # Write conformers to the output file
    with Chem.SDWriter(open(output_conformers, 'w')) as writer:
        for mol in conformers:
            writer.write(mol)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process ligand files and generate conformers.")
    parser.add_argument('in_ligands', action='store', type=str, help="Directory containing ligand files.")
    parser.add_argument('out_conformers', action='store', type=str, help="Output file for conformers.")
    args = parser.parse_args()

    # Disable RDKit logging
    disable_rdkit_logging()

    # Run the main function
    run(
        ligands_dir=args.in_ligands,
        output_conformers=args.out_conformers,
    )