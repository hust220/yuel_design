# File: extract_best_poses_parallel.py
import psycopg2
import re
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from psycopg2.extras import execute_batch
import sys
sys.path.append('../../')
from db_utils import db_connection

# TABLE_NAME = 'random_docking'
# ID_COLUMN = 'id'
# DOCKING_OUTPUT_COLUMN = 'docking_output'
# BEST_POSE_COLUMN = 'best_pose'
# BEST_SCORE_COLUMN = 'best_score'

TABLE_NAME = 'docking'
ID_COLUMN = 'id'
DOCKING_OUTPUT_COLUMN = 'docking_output'
BEST_POSE_COLUMN = 'best_pose'
BEST_SCORE_COLUMN = 'best_score'

# TABLE_NAME = 'medusadock_results'
# ID_COLUMN = 'molecule_id'
# DOCKING_OUTPUT_COLUMN = 'medusadock_pdb'
# BEST_POSE_COLUMN = 'best_pose'
# BEST_SCORE_COLUMN = 'medusadock_score'

def extract_poses(pdb_content):
    """Extract all poses from PDB content and find the one with lowest score"""
    if not pdb_content:
        return None, None
    
    if isinstance(pdb_content, bytes):
        try:
            pdb_content = pdb_content.decode('utf-8')
        except UnicodeDecodeError:
            return None, None
    
    # Split into individual poses
    current_pose = []
    in_pose = False
    current_score = None
    best_score = 9999
    best_pose = None
    for line in pdb_content.split('\n'):
        if line.startswith('REMARK POSE'):
            current_pose = [line]
            in_pose = True
        elif line.startswith('REMARK E_without_VDWR:'):
            current_score = float(line.split()[-1])
        elif line.startswith('ENDMDL'):
            if in_pose:
                current_pose.append(line)
                if current_score < best_score:
                    best_score = current_score
                    best_pose = '\n'.join(current_pose)
                current_pose = []
                in_pose = False
        elif in_pose:
            current_pose.append(line)

    return best_pose, best_score

def get_molecule_ids_to_process():
    """Get all molecule_ids that need processing"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT {ID_COLUMN} FROM {TABLE_NAME} 
                WHERE {DOCKING_OUTPUT_COLUMN} IS NOT NULL
                AND {BEST_POSE_COLUMN} IS NULL
                ORDER BY {ID_COLUMN}
            """)
            return [row[0] for row in cursor.fetchall()]

def get_pdb_data(molecule_id):
    """Get pdb data for a single molecule_id"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT {DOCKING_OUTPUT_COLUMN} FROM {TABLE_NAME} 
                WHERE {ID_COLUMN} = %s
            """, (molecule_id,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return row[0].tobytes().decode('utf-8')
                except Exception as e:
                    print(f"Error decoding data for {molecule_id}: {str(e)}")
            return None

def process_and_update(molecule_id):
    """Process a single molecule and update database with best pose"""
    pdb_data = get_pdb_data(molecule_id)
    if not pdb_data:
        return 0
    
    best_pose, best_score = extract_poses(pdb_data)
    if not best_pose:
        return 0
    
    # Update database within the same worker process
    with db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"""
                    UPDATE {TABLE_NAME}
                    SET {BEST_POSE_COLUMN} = %s,
                        {BEST_SCORE_COLUMN} = %s
                    WHERE {ID_COLUMN} = %s
                """, (best_pose.encode('utf-8'), best_score, molecule_id))
                conn.commit()
                return 1
            except Exception as e:
                conn.rollback()
                print(f"Failed to update {molecule_id}: {str(e)}")
                return 0

def process_batch(id_batch):
    """Process a batch of molecule IDs and update database"""
    count = 0
    for molecule_id in id_batch:
        count += process_and_update(molecule_id)
    return count

def ensure_best_pose_column():
    """Ensure the best_pose column exists in the table"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"""
                    ALTER TABLE {TABLE_NAME} 
                    ADD COLUMN IF NOT EXISTS {BEST_POSE_COLUMN} bytea
                """)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Error ensuring best_pose column exists: {str(e)}")
                raise

def main(batch_size=100, workers=16):
    """Main processing function"""
    ensure_best_pose_column()
    
    molecule_ids = get_molecule_ids_to_process()
    total = len(molecule_ids)
    if total == 0:
        print("No records to process - all best poses already extracted")
        return
    
    print(f"Starting to process {total} records with {workers} workers...")
    
    # Split IDs into batches
    id_batches = [molecule_ids[i:i + batch_size] 
                 for i in range(0, total, batch_size)]
    
    with Pool(workers) as pool:
        with tqdm(total=total, desc="Processing") as pbar:
            # Process batches in parallel and update counts
            for count in pool.imap_unordered(process_batch, id_batches):
                pbar.update(count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract best MedusaDock poses')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()

    main(args.batch, args.workers)
