#%%

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
import sys
import os
sys.path.append('../')
from db_utils import db_connection
from tqdm import tqdm

def create_diffsbdd_table(cursor):
    """Create the diffsbdd_generation table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diffsbdd_generation (
            id SERIAL PRIMARY KEY,
            molecule_name VARCHAR(255),
            sdf BYTEA,
            size INTEGER
        )
    """)

def process_sdf_file(file_path, cursor):
    """Process a single SDF file and store its molecules in the database."""
    # Get the ligand name from the filename (remove _generated.sdf)
    ligand_name = os.path.basename(file_path).replace('_generated.sdf', '')
    
    # Read the SDF file
    supplier = Chem.SDMolSupplier(file_path, sanitize=False, strictParsing=False)
    
    # Process each molecule in the SDF file
    for mol in supplier:
        if mol is not None:
            # Convert molecule to SDF bytes
            sdf_bytes = Chem.MolToMolBlock(mol).encode('utf-8')
            
            # Get the size (number of atoms)
            size = mol.GetNumAtoms()
            
            # Insert into database
            cursor.execute("""
                INSERT INTO diffsbdd_generation (molecule_name, sdf, size)
                VALUES (%s, %s, %s)
            """, (ligand_name, sdf_bytes, size))

def main():
    # Connect to the database
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Create the table
        create_diffsbdd_table(cursor)
        
        # Process all SDF files in the diffsbdd_generated folder
        sdf_folder = 'diffsbdd_generated'
        if not os.path.exists(sdf_folder):
            print(f"Error: {sdf_folder} directory does not exist")
            return
            
        for filename in tqdm(os.listdir(sdf_folder), desc="Processing SDF files"):
            if filename.endswith('_generated.sdf'):
                file_path = os.path.join(sdf_folder, filename)
                process_sdf_file(file_path, cursor)
        
        # Commit the changes
        conn.commit()
        print("Successfully stored all molecules in the database")
            

if __name__ == "__main__":
    main()


# %%
