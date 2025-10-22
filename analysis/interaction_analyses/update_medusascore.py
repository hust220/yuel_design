# File: extract_scores.py
import psycopg2
import re
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from psycopg2.extras import execute_batch
from contextlib import contextmanager
import time
import numpy as np

class ScoreExtractor:
    def __init__(self, db_params, batch_size=1000, workers=4):
        self.db_params = db_params
        self.batch_size = batch_size
        self.workers = workers
        self.all_ids = []

    @contextmanager
    def _connect(self):
        """Database connection with retry mechanism"""
        attempts = 0
        conn = None
        try:
            while True:
                try:
                    conn = psycopg2.connect(**self.db_params)
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

    def _extract_score(self, pdb_content):
        """Extract the minimum medusadock_score from PDB content"""
        if not pdb_content:
            return None
            
        if isinstance(pdb_content, bytes):
            try:
                pdb_content = pdb_content.decode('utf-8')
            except UnicodeDecodeError:
                return None
                
        scores = []
        pattern = r'REMARK E_without_VDWR:\s*(-?\d+(?:\.\d+)?)'
        
        for line in pdb_content.split('\n'):
            match = re.search(pattern, line)
            if match:
                try:
                    score = float(match.group(1))
                    scores.append(score)
                except (ValueError, TypeError):
                    continue
        
        return min(scores) if scores else None

    def _get_all_ids(self):
        """Get all molecule_ids that need processing"""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT molecule_id FROM medusadock_results 
                    WHERE medusadock_pdb IS NOT NULL
                    AND medusadock_score IS NULL
                    ORDER BY molecule_id
                """)
                self.all_ids = [row[0] for row in cursor.fetchall()]
                return len(self.all_ids)

    def _get_data_for_ids(self, id_batch):
        """Get pdb data for a batch of ids (called by worker processes)"""
        # Convert id_batch to list if it's a numpy array
        id_list = list(id_batch) if hasattr(id_batch, '__iter__') else [id_batch]
        
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT molecule_id, medusadock_pdb 
                    FROM medusadock_results 
                    WHERE molecule_id = ANY(%s)
                """, (id_list,))
                
                records = []
                for molecule_id, pdb_data in cursor:
                    if pdb_data is None:
                        continue
                    try:
                        pdb_str = pdb_data.tobytes().decode('utf-8')
                        records.append((molecule_id, pdb_str))
                    except Exception as e:
                        print(f"Error decoding data for {molecule_id}: {str(e)}")
                return records

    def _process_id_batch(self, id_batch):
        """Process a batch of IDs (called by worker processes)"""
        records = self._get_data_for_ids(id_batch)
        updates = []
        for molecule_id, pdb_data in records:
            score = self._extract_score(pdb_data)
            if score is not None:
                updates.append((score, molecule_id))
        return updates

    def _update_scores(self, updates):
        """Update scores in the database"""
        if not updates:
            return
            
        with self._connect() as conn:
            with conn.cursor() as cursor:
                try:
                    execute_batch(cursor, """
                        UPDATE medusadock_results
                        SET medusadock_score = %s
                        WHERE molecule_id = %s
                    """, updates)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"Failed to update scores: {str(e)}")
                    raise

    def run(self):
        """Main processing loop with parallel processing"""
        total = self._get_all_ids()
        if total == 0:
            print("No records to process - all scores already extracted")
            return
            
        print(f"Starting to process {total} records with {self.workers} workers...")
        
        # Split all IDs into chunks for parallel processing (using simple list slicing)
        id_batches = [self.all_ids[i:i + self.batch_size] 
                     for i in range(0, len(self.all_ids), self.batch_size)]
        
        with Pool(self.workers) as pool:
            with tqdm(total=total, desc="Processing") as pbar:
                # Process batches in parallel
                for updates in pool.imap_unordered(self._process_id_batch, id_batches):
                    if updates:
                        self._update_scores(updates)
                    pbar.update(len(updates))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract MedusaDock scores from PDB files')
    parser.add_argument('--batch', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    args = parser.parse_args()

    db_params = {
        'dbname': 'yuel_design',
        'user': 'juw1179',
        'host': 'submit03',
        'port': '5433'
    }

    extractor = ScoreExtractor(db_params, args.batch, args.workers)
    extractor.run()