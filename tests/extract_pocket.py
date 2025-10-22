#%%

import argparse
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser

def get_pocket(mol, pdb_path, threshold=6, ligand_match_threshold=0.5):
    """Identify protein residues within 6Å of the ligand, excluding the ligand itself if present in PDB"""
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []
    
    # Get ligand coordinates
    mol_atom_coords = mol.GetConformer().GetPositions()
    
    # First pass: collect all atoms and identify ligand atoms
    for ir, residue in enumerate(struct.get_residues()):
        for atom in residue.get_atoms():
            coord = atom.get_coord()
            # Check if this atom matches any ligand atom
            distances = np.linalg.norm(coord - mol_atom_coords, axis=1)
            if distances.min() > ligand_match_threshold:  # Not a ligand atom
                atom_coords.append(coord)
                residue_ids.append(ir)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(axis=1) <= threshold)[0]])

    return [r for (ir, r) in enumerate(struct.get_residues()) if ir in contact_residues]

def write_pocket_to_pdb(pocket_residues, output_path):
    """Write pocket residues to a properly formatted PDB file."""
    with open(output_path, 'w') as f:
        atom_serial = 1
        for residue in pocket_residues:
            resname = residue.get_resname()
            chain_id = residue.get_parent().id
            resseq = residue.id[1]
            icode = residue.id[2].strip() if residue.id[2] != ' ' else ''
            
            for atom in residue.get_atoms():
                name = atom.get_name()
                altloc = atom.get_altloc() if atom.get_altloc() != ' ' else ''
                coord = atom.get_coord()
                element = atom.element.strip().upper() if atom.element else name[0].strip().upper()

                line = (
                    f"ATOM  {atom_serial:5d} "
                    f"{name:<4}{altloc:1}"
                    f"{resname:>3} {chain_id:1}"
                    f"{resseq:4d}{icode:1}   "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}          "
                    f"{element:>2}\n"
                )
                f.write(line)
                atom_serial += 1

        f.write("END\n")

#%%

# pdb_path = '7ckz.pdb'
# mol_path = '7ckz_ligand.mol2'
# output_path = '7ckz_pocket.pdb'

pdb_path = '7e2y.pdb'
mol_path = '7e2y_ligand.sdf'
output_path = '7e2y_pocket.pdb'

# Load ligand molecule
if mol_path.endswith('.sdf'):
    mol = Chem.SDMolSupplier(mol_path, sanitize=False, strictParsing=False)[0]
elif mol_path.endswith('.mol2'):
    mol = Chem.MolFromMol2File(mol_path)
elif mol_path.endswith('.pdb'):
    mol = Chem.MolFromPDBFile(mol_path)
else:
    raise ValueError(f"Could not read ligand from {mol_path}")

# Get pocket residues
pocket_residues = get_pocket(mol, pdb_path)

# Save pocket to PDB file using custom writer
write_pocket_to_pdb(pocket_residues, output_path)
print(f"Pocket saved to {output_path}")
# %%
