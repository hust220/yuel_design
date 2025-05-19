# File: docking_worker.py
import psycopg2
import os
import argparse
from time import sleep
from tqdm import tqdm
import time
import tempfile
import subprocess
from pathlib import Path
import psycopg2.extras
from psycopg2.extras import execute_batch
import psycopg2
from psycopg2.extras import execute_values
import os
import sys
sys.path.append('../../')
from db_utils import db_connection

# docking_status: 0: pending, 1: docking, 2: success, 3: failed

class SlurmDockingWorker:
    def __init__(self, batch_size=50, n=None, table_name=None):
        self.batch_size = batch_size
        self.n = n
        self.table_name = table_name

    def _acquire_batch(self):
        """Atomically acquire a batch of tasks"""
        with db_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Get pending molecule IDs
                cursor.execute(f"""
                    SELECT id FROM {self.table_name}
                    WHERE docking_status = 0
                    ORDER BY id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """, (self.batch_size,))
                
                ids = [row[0] for row in cursor.fetchall()]
                if not ids:
                    return []
                
                # Mark as processing
                execute_batch(cursor, f"""
                    UPDATE {self.table_name}
                    SET docking_status = 1
                    WHERE id = %s
                """, [(i,) for i in ids])
                
                conn.commit()
                return ids
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def _process_molecule(self, mol_id):
        """Execute MedusaDock docking logic"""
        try:
            # Get database record
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT receptor, pocket, ligand, receptor_format, ligand_format, pocket_format 
                    FROM {self.table_name}
                    WHERE id = %s
                """, (mol_id,))
                r = cursor.fetchone()
                receptor_data = r[0].tobytes()
                pocket_data = r[1].tobytes()
                ligand_data = r[2].tobytes()
                receptor_format = r[3]
                ligand_format = r[4]
                pocket_format = r[5]

                cursor.close()

            if not receptor_data or not pocket_data or not ligand_data:
                raise ValueError(f"No data found for molecule {mol_id}")

            # Create temp directory
            docking_dir = Path(__file__).parent / 'docking'
            docking_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(docking_dir), prefix=f"dock_{mol_id}_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_path.mkdir(exist_ok=True)
                
                # Write receptor file with correct extension
                receptor_file = tmp_path / f"receptor.{receptor_format}"
                with open(receptor_file, 'wb') as f:
                    f.write(receptor_data)

                # Write ligand file with correct extension
                ligand_file = tmp_path / f"ligand.{ligand_format}"
                with open(ligand_file, 'wb') as f:
                    f.write(ligand_data)
                
                # Write pocket file with correct extension
                pocket_file = tmp_path / f"pocket.{pocket_format}"
                with open(pocket_file, 'wb') as f:
                    f.write(pocket_data)

                # Prepare output file
                output_file = tmp_path / "output.pdb"

                # path of medusadock.sh
                medusadock_sh = Path(__file__).parent / "medusadock.sh"
                # Execute MedusaDock command
                cmd = [
                    "bash", 
                    str(medusadock_sh),
                    str(receptor_file),
                    str(ligand_file),
                    str(pocket_file),
                    str(output_file)
                ]
                # Run command and check result
                result = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Verify output file
                if not output_file.exists():
                    raise RuntimeError("MedusaDock failed to generate output")
                
                # Read docking results
                with open(output_file, 'rb') as f:
                    output_data = f.read()

                return {
                    'docking_output': output_data,
                    'log': result.stdout,
                    'error_log': result.stderr
                }

        except subprocess.CalledProcessError as e:
            print(f"MedusaDock error for {mol_id}: {e.stderr}")
            return {
                'error': f"Process error (code {e.returncode}): {e.stderr[:200]}",
                'log': e.stdout
            }
        except Exception as e:
            print(f"General error processing {mol_id}: {str(e)}")
            return {
                'error': str(e)
            }

    def _update_results(self, results):
        """Update docking results in database"""
        with db_connection() as conn:
            cursor = conn.cursor()
            try:
                # Batch update successful records
                success_data = [
                    (res['docking_output'], mol_id)
                    for mol_id, res in results.items() 
                    if 'docking_output' in res
                ]
                
                if success_data:
                    execute_batch(cursor, f"""
                        UPDATE {self.table_name} 
                        SET 
                            docking_status = 2,
                            docking_output = %s
                        WHERE id = %s
                    """, [(pdb_data, mol_id) for pdb_data, mol_id in success_data])

                # Process failed records
                fail_data = [
                    (f"Error: {res.get('error', 'Unknown')}", mol_id)
                    for mol_id, res in results.items()
                    if 'error' in res
                ]
                
                if fail_data:
                    execute_batch(cursor, f"""
                        UPDATE {self.table_name} SET
                            docking_status = 3,
                            error_log = %s
                        WHERE id = %s
                    """, [(error_msg, mol_id) for error_msg, mol_id in fail_data])
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                print(f"Failed to update database: {str(e)}")
                raise
            finally:
                cursor.close()

    def run(self):
        """Worker main loop"""
        nbatches = 0
        while True:
            batch = self._acquire_batch()
            if not batch:
                break
            
            success = {}
            failed = []
            
            # Process current batch
            for mol_id in tqdm(batch, desc="Processing"):
                result = self._process_molecule(mol_id)
                if result:
                    success[mol_id] = result
                else:
                    failed.append(mol_id)
            
            # Update successful records
            if success:
                self._update_results(success)
                
            # Reset failed record status
            if failed:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    try:
                        execute_batch(cursor, f"""
                            UPDATE {self.table_name}
                            SET docking_status = 0
                            WHERE id = %s
                        """, [(i,) for i in failed])
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        print(f"Failed to reset failed records: {str(e)}")
                    finally:
                        cursor.close()
            
            nbatches += 1
            if self.n and nbatches >= self.n:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MedusaDock docking worker')
    parser.add_argument('--batch', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--n', type=int, help='Number of batches to process')
    parser.add_argument('--table', type=str, default='docking', help='Table name')
    args = parser.parse_args()

    worker = SlurmDockingWorker(
        batch_size=args.batch,
        n=args.n,
        table_name=args.table
    )
    worker.run()