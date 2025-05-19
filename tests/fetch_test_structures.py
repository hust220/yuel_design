from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
import sys
import os
sys.path.append('../')
from db_utils import db_connection

#%%
def get_unique_ligand_names():
    """Get unique names from the ligands table."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT pocket FROM random_docking")
            names = [row[0] for row in cursor.fetchall()]
    return names

def save_structures_for_pockets():
    """Save ligand and protein structures for each unique pocket."""
    # Create test_structures directory if it doesn't exist
    os.makedirs('test_structures', exist_ok=True)
    
    pockets = get_unique_ligand_names()
    print(f"Found {len(pockets)} unique pockets")
    
    with db_connection() as conn:
        with conn.cursor() as cursor:
            for pocket in pockets:
                # Get ligand structure
                cursor.execute("""
                    SELECT l.mol, l.protein_name 
                    FROM ligands l 
                    WHERE l.name = %s 
                    LIMIT 1
                """, (pocket,))
                ligand_data = cursor.fetchone()
                
                if ligand_data:
                    mol_bytes, protein_name = ligand_data
                    
                    # Save ligand structure
                    ligand_path = f'test_structures/{pocket}_pocket.mol'
                    with open(ligand_path, 'wb') as f:
                        f.write(mol_bytes.tobytes())
                    
                    # Get and save protein structure
                    cursor.execute("""
                        SELECT pdb 
                        FROM proteins 
                        WHERE name = %s
                    """, (protein_name,))
                    protein_data = cursor.fetchone()
                    
                    if protein_data:
                        pdb_bytes = protein_data[0].tobytes()
                        protein_path = f'test_structures/{pocket}_protein.pdb'
                        with open(protein_path, 'wb') as f:
                            f.write(pdb_bytes)
                        print(f"Saved structures for {pocket} (protein: {protein_name})")
                    else:
                        print(f"Warning: No protein structure found for {protein_name}")
                else:
                    print(f"Warning: No ligand structure found for {pocket}")

#%%
# Test the function
save_structures_for_pockets()
