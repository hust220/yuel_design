#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import sys
import os
sys.path.append('../..')
from db_utils import db_connection
import psycopg2
import io
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO
from multiprocessing import Pool
import re

def parse_pdb_residues(pdb_content):
    """
    Parse ATOM lines from a PDB file and group them by residue (chain_id, res_seq, res_name).
    Returns a dict: (chain_id, res_seq, res_name) -> list of atom dicts.
    """
    residues = {}
    for line in pdb_content.splitlines():
        if line.startswith("ATOM"):
            atom = {
                "line": line,
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "res_seq": int(line[22:26]),
                "chain_id": line[21],
                "res_name": line[17:20].strip(),
            }
            key = (atom["chain_id"], atom["res_seq"], atom["res_name"])
            residues.setdefault(key, []).append(atom)
    return residues

def get_pocket(protein_pdb_bytes, ligand_mol_bytes):
    """Extract pocket residues and return a PDB file as bytes, without using Bio.PDB."""
    # Parse protein PDB
    pdb_content = protein_pdb_bytes.decode('utf-8')
    residues = parse_pdb_residues(pdb_content)

    # Parse ligand molblock
    mol_block = ligand_mol_bytes.decode('utf-8') if isinstance(ligand_mol_bytes, bytes) else ligand_mol_bytes
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, strictParsing=False)
    ligand_coords = mol.GetConformer().GetPositions()

    # Find residues within 6Å of any ligand atom
    pocket_residues = set()
    for key, atoms in residues.items():
        atom_coords = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms])
        dists = np.linalg.norm(atom_coords[:, None, :] - ligand_coords[None, :, :], axis=-1)
        min_dist = np.min(dists)
        if min_dist <= 6.0:
            pocket_residues.add(key)

    # Write only ATOM lines for pocket residues
    pocket_lines = [atom["line"] for key, atoms in residues.items() if key in pocket_residues for atom in atoms]
    pocket_pdb_str = "\n".join(pocket_lines) + "\n"
    return pocket_pdb_str.encode('utf-8')

def create_table():
    """Create the pockets table if it does not exist."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pockets (
                    id serial PRIMARY KEY,
                    ligand_id integer NOT NULL,
                    protein_id integer NOT NULL,
                    pdb bytea NOT NULL,
                    created_at timestamptz DEFAULT now()
                );
            """)
            conn.commit()

def get_ligand_ids():
    # make sure ligand id doesn't exist in the pockets table.
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM ligands")
            ids = [row[0] for row in cur.fetchall()]
            cur.execute("SELECT ligand_id FROM pockets")
            existing_ids = [row[0] for row in cur.fetchall()]
            ids = [id for id in ids if id not in existing_ids]
    return ids

def process_ligand(ligand_id):
    """Process a single ligand by its ID and insert pocket info into the database."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT l.id, l.name, l.mol, p.id, p.pdb
                FROM ligands l
                JOIN proteins p ON l.protein_name = p.name
                WHERE l.id = %s
            """, (ligand_id,))
            row = cur.fetchone()
            if row is None:
                print(f"Ligand ID {ligand_id} not found or no matching protein.")
                return
            ligand_id, ligand_name, mol_bytes, protein_id, pdb_bytes = row
            try:
                mol_bytes = mol_bytes.tobytes()
                pdb_bytes = pdb_bytes.tobytes()

                pocket_pdb = get_pocket(pdb_bytes, mol_bytes)
                upload_pocket(pocket_pdb, ligand_id, protein_id)
                
            except Exception as e:
                print(f"Failed to process ligand {ligand_name} (ID {ligand_id}): {e}")

def get_pockets(num_workers=1):
    """Parallelize pocket extraction over all ligands."""
    ids = get_ligand_ids()
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_ligand, ids), total=len(ids), desc='Processing ligands'))

def upload_pocket(pocket_pdb_bytes, ligand_id, protein_id):
    """Insert the pocket PDB bytes into the pockets table."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pockets (protein_id, ligand_id, pdb)
                VALUES (%s, %s, %s)
            """, (protein_id, ligand_id, psycopg2.Binary(pocket_pdb_bytes)))
            conn.commit()

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload pockets to the database in parallel.")
    parser.add_argument('num_workers', type=int, nargs='?', default=1, help='Number of worker processes for parallel processing.')
    args = parser.parse_args()

    create_table()
    get_pockets(args.num_workers)
