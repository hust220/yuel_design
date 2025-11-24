import os
import sys
import re
from pathlib import Path
from tqdm import tqdm

# Add the path to import db_utils from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from db_utils import create_connection

def create_tables(conn):
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moad_proteins (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            pdb TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moad_ligands (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            protein_name VARCHAR(100) NOT NULL,
            residue_name VARCHAR(10) NOT NULL,
            residue_id VARCHAR(10) NOT NULL,
            chain_id VARCHAR(10) NOT NULL,
            mol TEXT NOT NULL,
            FOREIGN KEY (protein_name) REFERENCES moad_proteins(name)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moad_pockets (
            id SERIAL PRIMARY KEY,
            name VARCHAR(150) NOT NULL,
            ligand_name VARCHAR(150) NOT NULL,
            protein_name VARCHAR(100) NOT NULL,
            pdb TEXT NOT NULL,
            FOREIGN KEY (protein_name) REFERENCES moad_proteins(name),
            FOREIGN KEY (ligand_name) REFERENCES moad_ligands(name)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moad_proteins_name ON moad_proteins(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moad_ligands_protein_name ON moad_ligands(protein_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moad_ligands_name ON moad_ligands(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moad_pockets_protein_name ON moad_pockets(protein_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moad_pockets_name ON moad_pockets(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moad_pockets_ligand_name ON moad_pockets(ligand_name)")
    
    conn.commit()
    print("Database tables created successfully")

def store_protein(conn, name, pdb_content):
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO moad_proteins (name, pdb) 
            VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET
            pdb = EXCLUDED.pdb
        """, (name, pdb_content))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing protein {name}: {e}")
        conn.rollback()
        return False

def store_ligand(conn, name, protein_name, residue_name, residue_id, chain_id, mol_content):
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO moad_ligands (name, protein_name, residue_name, residue_id, chain_id, mol) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (name, protein_name, residue_name, residue_id, chain_id, mol_content))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing ligand {name}: {e}")
        conn.rollback()
        return False

def store_pocket(conn, name, ligand_name, protein_name, pdb_content):
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO moad_pockets (name, ligand_name, protein_name, pdb) 
            VALUES (%s, %s, %s, %s)
        """, (name, ligand_name, protein_name, pdb_content))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing pocket {name}: {e}")
        conn.rollback()
        return False

def process_proteins(conn, proteins_dir):
    proteins_path = Path(proteins_dir)
    if not proteins_path.exists():
        print(f"Proteins directory not found: {proteins_dir}")
        return set()
    
    processed_proteins = set()
    protein_files = list(proteins_path.glob("*_protein.pdb"))
    
    print(f"Processing {len(protein_files)} protein files...")
    
    for pdb_file in tqdm(protein_files, desc="Processing proteins"):
        try:
            protein_name = pdb_file.stem.replace("_protein", "")
            
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
            
            if store_protein(conn, protein_name, pdb_content):
                processed_proteins.add(protein_name)
            
        except Exception as e:
            print(f"Error processing protein file {pdb_file}: {e}")
    
    print(f"Successfully processed {len(processed_proteins)} proteins")
    return processed_proteins

def process_ligands(conn, ligands_dir, processed_proteins):
    ligands_path = Path(ligands_dir)
    if not ligands_path.exists():
        print(f"Ligands directory not found: {ligands_dir}")
        return set()
    
    ligand_files = list(ligands_path.glob("*.mol"))
    processed_ligands = set()
    skipped_ligands = 0
    
    print(f"Processing {len(ligand_files)} ligand files...")
    
    for mol_file in tqdm(ligand_files, desc="Processing ligands"):
        try:
            file_stem = mol_file.stem
            parts = file_stem.split('_')
            if len(parts) >= 4:
                # Format: {pdb_code}_{residue_name}{residue_id}_{chain_id}_{i}.mol
                protein_name = parts[0]
                residue_name_id = parts[1]
                chain_id = parts[2]
                ligand_index = parts[3]
                
                # Extract residue_name and residue_id from residue_name_id
                match = re.match(r'([A-Za-z]+)(\d+.*)', residue_name_id)
                if match:
                    residue_name = match.group(1)
                    residue_id = match.group(2)
                else:
                    residue_name = residue_name_id
                    residue_id = ""
                
                ligand_name = file_stem
            else:
                print(f"Warning: Unexpected ligand filename format: {mol_file}")
                continue
            
            if protein_name not in processed_proteins:
                print(f"Warning: Protein {protein_name} not found for ligand {ligand_name}")
                skipped_ligands += 1
                continue
            
            with open(mol_file, 'r') as f:
                mol_content = f.read()
            
            if store_ligand(conn, ligand_name, protein_name, residue_name, residue_id, chain_id, mol_content):
                processed_ligands.add(ligand_name)
            
        except Exception as e:
            print(f"Error processing ligand file {mol_file}: {e}")
    
    print(f"Successfully processed {len(processed_ligands)} ligands")
    print(f"Skipped {skipped_ligands} ligands (no corresponding protein)")
    return processed_ligands

def process_pockets(conn, pockets_dir, processed_proteins, processed_ligands):
    pockets_path = Path(pockets_dir)
    if not pockets_path.exists():
        print(f"Pockets directory not found: {pockets_dir}")
        return
    
    pocket_files = list(pockets_path.glob("*_pocket.pdb"))
    processed_pockets = 0
    skipped_pockets = 0
    
    print(f"Processing {len(pocket_files)} pocket files...")
    
    for pdb_file in tqdm(pocket_files, desc="Processing pockets"):
        try:
            file_stem = pdb_file.stem
            base = file_stem.replace("_pocket", "")
            parts = base.split('_')
            if len(parts) >= 4:
                # Format base: {pdb_code}_{residue_name}{residue_id}_{chain_id}_{i}
                protein_name = parts[0]
                ligand_name = base  # ligand file stem equals base
                pocket_name = file_stem
            else:
                print(f"Warning: Unexpected pocket filename format: {pdb_file}")
                continue
            
            if protein_name not in processed_proteins:
                print(f"Warning: Protein {protein_name} not found for pocket {pocket_name}")
                skipped_pockets += 1
                continue
            
            if ligand_name not in processed_ligands:
                print(f"Warning: Ligand {ligand_name} not found for pocket {pocket_name}")
                skipped_pockets += 1
                continue
            
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
            
            if store_pocket(conn, pocket_name, ligand_name, protein_name, pdb_content):
                processed_pockets += 1
        except Exception as e:
            print(f"Error processing pocket file {pdb_file}: {e}")
    
    print(f"Successfully processed {processed_pockets} pockets")
    print(f"Skipped {skipped_pockets} pockets (no corresponding protein or ligand)")

def get_database_stats(conn):
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM moad_proteins")
    protein_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM moad_ligands")
    ligand_count = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT protein_name, COUNT(*) as ligand_count 
        FROM moad_ligands 
        GROUP BY protein_name 
        ORDER BY ligand_count DESC 
        LIMIT 10
    """)
    top_proteins = cursor.fetchall()
    
    cursor.execute("""
        SELECT residue_name, COUNT(*) as count 
        FROM moad_ligands 
        GROUP BY residue_name 
        ORDER BY count DESC 
        LIMIT 10
    """)
    top_residues = cursor.fetchall()
    
    print(f"\nDatabase Statistics:")
    print(f"Total proteins: {protein_count}")
    print(f"Total ligands: {ligand_count}")
    print(f"Average ligands per protein: {ligand_count/protein_count:.2f}" if protein_count > 0 else "")
    
    print(f"\nTop 10 proteins by ligand count:")
    for protein_name, count in top_proteins:
        print(f"  {protein_name}: {count} ligands")
    
    print(f"\nTop 10 residue types:")
    for residue_name, count in top_residues:
        print(f"  {residue_name}: {count} ligands")

if __name__ == '__main__':
    proteins_dir = "/home/tyq4zn/scratch/datasets/MOAD/proteins"
    ligands_dir = "/home/tyq4zn/scratch/datasets/MOAD/ligands"
    pockets_dir = "/home/tyq4zn/scratch/datasets/MOAD/pockets"
    
    print("MOAD Database Initialization")
    print("="*50)

    conn = create_connection()
    
    try:
        print("\nStep 1: Creating database tables...")
        create_tables(conn)
        
        print("\nStep 2: Processing protein files...")
        processed_proteins = process_proteins(conn, proteins_dir)
        
        print("\nStep 3: Processing ligand files...")
        processed_ligands = process_ligands(conn, ligands_dir, processed_proteins)
        
        print("\nStep 4: Processing pocket files...")
        process_pockets(conn, pockets_dir, processed_proteins, processed_ligands)
        
        print("\nStep 5: Database statistics...")
        get_database_stats(conn)
        
        print("\n" + "="*50)
        print("INITIALIZATION COMPLETE")
        print("="*50)
    
    finally:
        conn.close()
    