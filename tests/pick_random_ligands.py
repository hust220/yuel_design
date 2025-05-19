#%%
import pickle
import random
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
import time
from contextlib import contextmanager
from tqdm import tqdm

# Database connection parameters
DB_PARAMS = {
    'dbname': 'yuel_design',
    'user': 'juw1179',
    'host': 'submit03',
    'port': '5433'
}

@contextmanager
def db_connect():
    """Database connection with retry mechanism"""
    attempts = 0
    conn = None
    try:
        while True:
            try:
                conn = psycopg2.connect(**DB_PARAMS)
                conn.autocommit = False
                break
            except psycopg2.OperationalError as e:
                if attempts < 10:
                    time.sleep(2 ** attempts)
                    attempts += 1
                    continue
                else:
                    raise
        yield conn
    finally:
        if conn:
            conn.close()

def get_molecule_names_from_pkl(pkl_file_path):
    """Read .pkl file and extract molecule names"""
    with open(pkl_file_path, 'rb') as f:
        data_list = pickle.load(f)
    return [item['molecule'] for item in data_list]

def create_random_docking_table():
    with db_connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS random_docking (
                    id SERIAL PRIMARY KEY,
                    receptor TEXT NOT NULL,
                    ligand TEXT NOT NULL,
                    pocket TEXT,
                    ligand_size INTEGER NOT NULL,
                    docking_status INTEGER DEFAULT 0 NOT NULL,
                    docking_output BYTEA,
                    best_score FLOAT,
                    best_pose BYTEA,
                    error_log TEXT
                )
            """)
            conn.commit()

def get_random_ligands(pocket_name, min_size=11, max_size=30):
    """Get one random ligand for each size between min_size and max_size, excluding the receptor name"""
    ligand_names = []
    
    with db_connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT size FROM ligands 
                WHERE name = %s
            """, (pocket_name,))
            result = cursor.fetchone()
            if result:
                ligand_names.append((int(result[0]), pocket_name))
            else:
                print(f"Warning: Pocket {pocket_name} not found in ligands table")
            
            for size in range(min_size, max_size + 1):
                cursor.execute("""
                    SELECT name FROM ligands 
                    WHERE size = %s
                    AND name != %s
                    ORDER BY random()
                    LIMIT 1
                """, (size, pocket_name))
                result = cursor.fetchone()
                if result:
                    ligand_names.append((size, result[0]))
                else:
                    print(f"Warning: No suitable ligands with size {size} found for pocket {pocket_name}")
    return ligand_names

def insert_docking_data(receptor_name, pocket_name, ligand_names):
    with db_connect() as conn:
        with conn.cursor() as cursor:
            data = [(receptor_name, pocket_name, ligand_name, size) for size, ligand_name in ligand_names]
            
            execute_batch(cursor, """
                INSERT INTO random_docking (receptor, pocket, ligand, ligand_size)
                VALUES (%s, %s, %s, %s)
            """, data)
            conn.commit()

def process_receptors(pkl_file_path):
    molecule_names = get_molecule_names_from_pkl(pkl_file_path)
    create_random_docking_table()
    
    for molecule_name in tqdm(molecule_names, total=len(molecule_names), desc="Processing receptors"):
        receptor_name = molecule_name.split('_')[0]
        pocket_name = molecule_name
        try:
            ligand_names = get_random_ligands(pocket_name)
            
            if ligand_names:
                insert_docking_data(receptor_name, pocket_name, ligand_names)
                # print(f"Inserted {len(ligand_names)} ligands for receptor {receptor_name}")
            else:
                print(f"Warning: No suitable ligands found for receptor {receptor_name}")
        
        except Exception as e:
            print(f"Error processing receptor {receptor_name}: {str(e)}")
            raise e
            # continue

#%%
if __name__ == "__main__":
    # Replace with your actual pkl file path
    pkl_file_path = "../datasets/MOAD_test.pkl"
    process_receptors(pkl_file_path)

#%%


