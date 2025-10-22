# File: calculate_docking_scores.py
import psycopg2
import re
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from psycopg2.extras import execute_batch
import time
import numpy as np
import sys
sys.path.append('../../')
from db_utils import db_connection

class ScoreExtractor:
    def __init__(self, batch_size=1000, workers=4):
        self.batch_size = batch_size
        self.workers = workers
        self.all_ids = []

    def _extract_score(self, docking_output):
        """Extract the minimum docking score from docking output"""
        if not docking_output:
            return None
            
        if isinstance(docking_output, bytes):
            try:
                docking_output = docking_output.decode('utf-8')
            except UnicodeDecodeError:
                return None
                
        scores = []
        # Pattern to match docking scores - adjust this based on your docking output format
        pattern = r'REMARK E_without_VDWR:\s*(-?\d+(?:\.\d+)?)'
        
        for line in docking_output.split('\n'):
            match = re.search(pattern, line)
            if match:
                try:
                    score = float(match.group(1))
                    scores.append(score)
                except (ValueError, TypeError):
                    continue
        
        return min(scores) if scores else None

    def _get_all_ids(self):
        """Get all IDs that need processing"""
        with db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM docking 
                    WHERE docking_output IS NOT NULL
                    AND best_score IS NULL
                    AND docking_status = 2
                    ORDER BY id
                """)
                self.all_ids = [row[0] for row in cursor.fetchall()]
                return len(self.all_ids)

    def _get_data_for_ids(self, id_batch):
        """Get docking output data for a batch of ids"""
        id_list = list(id_batch) if hasattr(id_batch, '__iter__') else [id_batch]
        
        with db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, docking_output 
                    FROM docking 
                    WHERE id = ANY(%s)
                """, (id_list,))
                
                records = []
                for id, output_data in cursor:
                    if output_data is None:
                        continue
                    try:
                        output_str = output_data.tobytes().decode('utf-8')
                        records.append((id, output_str))
                    except Exception as e:
                        print(f"Error decoding data for id {id}: {str(e)}")
                return records

    def _process_id_batch(self, id_batch):
        """Process a batch of IDs"""
        records = self._get_data_for_ids(id_batch)
        updates = []
        for id, output_data in records:
            score = self._extract_score(output_data)
            if score is not None:
                updates.append((score, id))
        return updates

    def _update_scores(self, updates):
        """Update scores in the database"""
        if not updates:
            return
            
        with db_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    execute_batch(cursor, """
                        UPDATE docking
                        SET best_score = %s
                        WHERE id = %s
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
        
        # Split all IDs into chunks for parallel processing
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
    parser = argparse.ArgumentParser(description='Extract docking scores from docking output')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()

    extractor = ScoreExtractor(args.batch, args.workers)
    extractor.run()