#%%

import os
import re
from tqdm import tqdm
from rdkit import Chem
import psycopg2
from psycopg2 import sql
import psycopg2.extras
from contextlib import closing
from rdkit import RDLogger
from io import StringIO

RDLogger.DisableLog('rdApp.warning')

def initialize_database(db_params):
    """Initialize the PostgreSQL database with optimized settings."""
    conn = psycopg2.connect(**db_params)
    
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            id SERIAL PRIMARY KEY,
            ligand_name TEXT NOT NULL,
            sdf BYTEA NOT NULL,
            mol2 BYTEA,
            size INTEGER NOT NULL CHECK (size > 0),
            seed INTEGER,
            nattempts INTEGER,
            nattempts_invalid INTEGER,
            nattempts_unconnected INTEGER,
            nattempts_large_rings INTEGER,
            duration REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            medusadock_status INTEGER
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS proteins (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            pdb BYTEA NOT NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ligands (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            protein_name TEXT NOT NULL,
            mol BYTEA NOT NULL,
            sdf BYTEA,
            mol2 BYTEA,
            size INTEGER NOT NULL CHECK (size > 0)
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS medusadock_results (
            molecule_id INTEGER PRIMARY KEY,
            medusadock_pdb BYTEA,
            medusadock_score DOUBLE PRECISION,
            error_log TEXT
        );
        """)
    conn.commit()
    return conn

def scan_folder(folder_path):
    sdf_files = [file for file in os.listdir(folder_path) if file.endswith(".sdf")]
    print('Found', len(sdf_files), 'sdf files')
    for sdf_file in tqdm(sdf_files, total=len(sdf_files)):
        m = re.match(r"(.+)_size(\d+)_n(\d+)_s(\d+)\.sdf", sdf_file)
        if m:
            name = m.group(1)
            size = int(m.group(2))
            nsamples = int(m.group(3))
            seed = int(m.group(4))
            
            block = ""
            for line in open(os.path.join(folder_path, sdf_file)):
                block += line
                if line.startswith('$$$$'):
                    yield block, name, size, 1, seed
                    block = ""

def scan_sdf(sdf_file):
    block = ""
    iline = 0
    name = None
    size = 0
    for line in open(sdf_file):
        if iline == 0:
            name = line.strip()
        elif iline == 3:
            ls = line.strip().split()
            if len(ls) == 11:
                size = int(ls[0])                
        block += line
        iline += 1
        if line.startswith('$$$$'):
            yield block, name, size, 1, None
            block = ""
            iline = 0

def insert_into_database(conn, sdf, name, size, seed=None):
    sdf = sdf.encode("utf-8")
    with conn.cursor() as cursor:
        if seed is None:
            cursor.execute(
                "INSERT INTO molecules (ligand_name, size, sdf) VALUES (%s, %s, %s)",
                (name, size, sdf)
            )
        else:
            cursor.execute(
                "INSERT INTO molecules (ligand_name, size, seed, sdf) VALUES (%s, %s, %s, %s)",
                (name, size, seed, sdf)
            )
    conn.commit()

def insert_batch(conn, batch):
    """Insert a batch of records into the database."""
    with conn.cursor() as cursor:
        psycopg2.extras.execute_batch(cursor, """
            INSERT INTO molecules (ligand_name, size, seed, sdf)
            VALUES (%s, %s, %s, %s)
        """, batch)
    conn.commit()

def load_molecules(conn, test_folder):
    batch = []
    BATCH_SIZE = 1000

    try:
        for sdf_block, name, size, nsamples, seed in scan_folder(test_folder):
            batch.append((
                name,
                int(size),
                int(seed) if seed is not None else None,
                sdf_block.encode('utf-8')
            ))
            
            if len(batch) >= BATCH_SIZE:
                insert_batch(conn, batch)
                batch = []

        if batch:
            insert_batch(conn, batch)
            
    finally:
        conn.close()

def load_proteins(conn, folder):
    """Load PDB files from folder into proteins table in batches"""
    batch = []
    BATCH_SIZE = 10

    for pdb_file in tqdm(os.listdir(folder), desc="Loading proteins"):
        if not pdb_file.endswith('.pdb'):
            continue
            
        try:
            name = os.path.splitext(pdb_file)[0].split('_')[0]
            with open(os.path.join(folder, pdb_file), 'rb') as f:
                pdb_data = f.read()
            
            batch.append((name, pdb_data))
            
            if len(batch) >= BATCH_SIZE:
                with conn.cursor() as cursor:
                    psycopg2.extras.execute_batch(cursor, """
                        INSERT INTO proteins (name, pdb) VALUES (%s, %s)
                    """, batch)
                conn.commit()
                batch = []
                
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            continue
    
    if batch:
        with conn.cursor() as cursor:
            psycopg2.extras.execute_batch(cursor, """
                INSERT INTO proteins (name, pdb) VALUES (%s, %s)
            """, batch)
        conn.commit()

def mol_size(block):
    iline = 0
    for line in block.split('\n'):
        if iline == 3:
            ls = line.strip().split()
            if len(ls) == 11:
                return int(ls[0])
        iline += 1
    return None

def load_ligands(conn, folder):
    batch = []
    BATCH_SIZE = 100
    
    mol_files = [f for f in os.listdir(folder) if f.endswith('.mol')]
    
    for mol_file in tqdm(mol_files, desc="Loading ligands"):
        try:
            ligand_name = os.path.splitext(mol_file)[0]
            protein_name = ligand_name.split('_')[0]
            
            with open(os.path.join(folder, mol_file), 'r') as f:
                mol_data = f.read()
            
            size = mol_size(mol_data)
            mol_data = mol_data.encode('utf-8')
            batch.append((ligand_name, protein_name, mol_data, size))
            
            if len(batch) >= BATCH_SIZE:
                with conn.cursor() as cursor:
                    psycopg2.extras.execute_batch(cursor, """
                        INSERT INTO ligands 
                        (name, protein_name, mol, size) 
                        VALUES (%s, %s, %s, %s)
                    """, batch)
                conn.commit()
                batch = []
                
        except Exception as e:
            print(f"Error processing {mol_file}: {str(e)}")
            continue
    
    if batch:
        with conn.cursor() as cursor:
            psycopg2.extras.execute_batch(cursor, """
                INSERT INTO ligands 
                (name, protein_name, mol, size) 
                VALUES (%s, %s, %s, %s)
            """, batch)
        conn.commit()

#%%

if __name__ == '__main__':
    # PostgreSQL connection parameters
    db_params = {
        'dbname': 'yuel_design',
        'user': 'juw1179',
        # 'password': 'your_password',
        'host': 'submit03',
        'port': '5433'
    }
    
    test_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.dirname(test_folder)
    moad_test_folder = os.path.join(test_folder, 'MOAD_test')
    proteins_folder = os.path.join(root_folder, 'data', 'MOAD', 'processed', 'proteins')
    ligands_folder = os.path.join(root_folder, 'data', 'MOAD', 'processed', 'ligands')
    
    conn = initialize_database(db_params)

    # load_molecules(conn, moad_test_folder)
    # load_proteins(conn, proteins_folder)
    load_ligands(conn, ligands_folder)

    conn.close()

# %%
