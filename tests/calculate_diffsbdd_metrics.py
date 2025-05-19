from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Descriptors, QED
import sys
import os
sys.path.append('../')
from db_utils import db_connection
from tqdm import tqdm
import networkx as nx
from src import sascorer
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

def is_connected(rdkit_mol):
    G = nx.Graph()
    
    for atom in rdkit_mol.GetAtoms():
        G.add_node(atom.GetIdx())
    
    for bond in rdkit_mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    return nx.is_connected(G)

def calculate_qed(mol):
    try:
        return QED.default(mol)
    except:
        return None

def calculate_sas(mol):
    try:
        return sascorer.calculateScore(mol)
    except:
        return None

def calculate_lipinski(mol):
    try:
        passes_ro5 = all([
            Descriptors.MolWt(mol) <= 500,
            Descriptors.MolLogP(mol) <= 5,
            Lipinski.NumHDonors(mol) <= 5,
            Lipinski.NumHAcceptors(mol) <= 10
        ])
        return passes_ro5
    except:
        return None

def has_large_rings(mol):
    try:
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 8:  # Consider rings with more than 8 atoms as large
                return True
        return False
    except:
        return None

def add_metrics_columns(cursor):
    """Add metric columns to the diffsbdd_generation table if they don't exist."""
    cursor.execute("""
        ALTER TABLE diffsbdd_generation 
        ADD COLUMN IF NOT EXISTS validity BOOLEAN,
        ADD COLUMN IF NOT EXISTS connectivity BOOLEAN,
        ADD COLUMN IF NOT EXISTS large_ring_rate BOOLEAN,
        ADD COLUMN IF NOT EXISTS qed FLOAT,
        ADD COLUMN IF NOT EXISTS sas FLOAT,
        ADD COLUMN IF NOT EXISTS lipinski BOOLEAN
    """)

def analyze_molecules():
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Add metric columns if they don't exist
        add_metrics_columns(cursor)
        
        # Get all molecules from the database
        cursor.execute("SELECT id, sdf FROM diffsbdd_generation")
        rows = cursor.fetchall()
        
        for row in tqdm(rows, desc="Analyzing molecules"):
            mol_id, sdf_string = row
            
            # Convert SDF string to molecule
            mol = Chem.MolFromMolBlock(sdf_string, sanitize=False)
            if mol is None:
                continue
                
            # Calculate metrics
            validity = is_valid(mol)
            connectivity = is_connected(mol) if validity else False
            large_ring_rate = has_large_rings(mol) if validity else False
            qed = calculate_qed(mol) if validity else None
            sas = calculate_sas(mol) if validity else None
            lipinski = calculate_lipinski(mol) if validity else None
            
            # Update database with metrics
            cursor.execute("""
                UPDATE diffsbdd_generation 
                SET validity = %s,
                    connectivity = %s,
                    large_ring_rate = %s,
                    qed = %s,
                    sas = %s,
                    lipinski = %s
                WHERE id = %s
            """, (validity, connectivity, large_ring_rate, qed, sas, lipinski, mol_id))
            
        conn.commit()

if __name__ == "__main__":
    analyze_molecules() 