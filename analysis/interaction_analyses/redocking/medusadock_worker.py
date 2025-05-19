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
from contextlib import contextmanager

# docking_status: 0: pending, 1: docking, 2: success, 3: failed

class SlurmDockingWorker:
    def __init__(self, db_params, batch_size=50, n=None):
        self.db_params = db_params
        self.batch_size = batch_size
        self.n = n

    @contextmanager
    def _connect(self):
        """Database connection with retry mechanism"""
        attempts = 0
        conn = None
        try:
            while True:
                try:
                    conn = psycopg2.connect(**self.db_params)
                    conn.autocommit = False  # We'll manage transactions manually
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

    def _acquire_batch(self):
        """Atomically acquire a batch of tasks"""
        with self._connect() as conn:
            try:
                # PostgreSQL uses BEGIN for transactions
                cursor = conn.cursor()
                
                # Get pending molecule IDs
                cursor.execute("""
                    SELECT id FROM molecules
                    WHERE medusadock_status = 0
                    ORDER BY id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """, (self.batch_size,))
                
                ids = [row[0] for row in cursor.fetchall()]
                if not ids:
                    return []
                
                # Mark as processing
                execute_batch(cursor, """
                    UPDATE molecules
                    SET medusadock_status = 1
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
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ligand_name, sdf FROM molecules
                    WHERE id = %s
                """, (mol_id,))
                molecule = cursor.fetchone()

                ligand_name = molecule[0]
                cursor.execute("""
                    SELECT mol, protein_name FROM ligands
                    WHERE name = %s
                """, (ligand_name,))
                ligand_row = cursor.fetchone()

                protein_name = ligand_row[1]
                cursor.execute("""
                    SELECT pdb FROM proteins
                    WHERE name = %s
                """, (protein_name,))
                protein_row = cursor.fetchone()

                receptor_pdb = protein_row[0]
                pocket_mol = ligand_row[0]
                ligand_sdf = molecule[1]
                cursor.close()

            if not receptor_pdb or not pocket_mol or not ligand_sdf:
                raise ValueError(f"No data found for molecule {mol_id}")

            # Create temp directory
            docking_dir = Path(__file__).parent / 'docking'
            docking_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(docking_dir), prefix=f"dock_{mol_id}_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_path.mkdir(exist_ok=True)
                
                # Write receptor file
                receptor_file = tmp_path / "receptor.pdb"
                with open(receptor_file, 'wb') as f:
                    f.write(receptor_pdb.tobytes() if hasattr(receptor_pdb, 'tobytes') else receptor_pdb)

                # Write ligand file
                ligand_file = tmp_path / "ligand.sdf"
                with open(ligand_file, 'wb') as f:
                    f.write(ligand_sdf.tobytes() if hasattr(ligand_sdf, 'tobytes') else ligand_sdf)
                
                # Write pocket file
                pocket_file = tmp_path / "pocket.mol"
                with open(pocket_file, 'wb') as f:
                    f.write(pocket_mol.tobytes() if hasattr(pocket_mol, 'tobytes') else pocket_mol)

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
                    'medusadock_pdb': output_data,
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
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                # Batch update successful records
                success_data = [
                    (res['medusadock_pdb'], mol_id)
                    for mol_id, res in results.items() 
                    if 'medusadock_pdb' in res
                ]
                
                if success_data:
                    # First update the molecules table
                    execute_batch(cursor, """
                        UPDATE molecules SET
                            medusadock_status = 2
                        WHERE id = %s
                    """, [(mol_id,) for _, mol_id in success_data])

                    # Then update medusadock_results
                    execute_batch(cursor, """
                        INSERT INTO medusadock_results (molecule_id, medusadock_pdb)
                        VALUES (%s, %s)
                        ON CONFLICT (molecule_id) 
                        DO UPDATE SET medusadock_pdb = EXCLUDED.medusadock_pdb
                    """, [(mol_id, pdb_data) for pdb_data, mol_id in success_data])

                # Process failed records
                fail_data = [
                    (f"Error: {res.get('error', 'Unknown')}", mol_id)
                    for mol_id, res in results.items()
                    if 'error' in res
                ]
                
                if fail_data:
                    # For failed docking attempts
                    # First update the molecules table
                    execute_batch(cursor, """
                        UPDATE molecules SET
                            medusadock_status = 3
                        WHERE id = %s
                    """, [(mol_id,) for _, mol_id in fail_data])

                    # Then update medusadock_results
                    for error_msg, mol_id in fail_data:
                        cursor.execute("""
                            INSERT INTO medusadock_results (molecule_id, error_log)
                            VALUES (%s, %s)
                            ON CONFLICT (molecule_id) 
                            DO UPDATE SET error_log = COALESCE(medusadock_results.error_log, '') || EXCLUDED.error_log
                        """, (mol_id, error_msg))

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
                with self._connect() as conn:
                    cursor = conn.cursor()
                    try:
                        execute_batch(cursor, """
                            UPDATE molecules
                            SET medusadock_status = 0
                            WHERE id = %s
                        """, [(i,) for i in failed])
                        conn.commit()
                    finally:
                        cursor.close()

            nbatches += 1
            if self.n is not None and nbatches >= self.n:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--db", required=True, help="PostgreSQL connection string")
    parser.add_argument("--batch", type=int, default=20, 
                       help="Molecules per batch")
    parser.add_argument("--n", type=int, default=1, help="Number of batches to process")
    args = parser.parse_args()

    db_params = {
        'dbname': 'yuel_design',
        'user': 'juw1179',
        # 'password': 'your_password',
        'host': 'submit03',
        'port': '5433'
    }

    worker = SlurmDockingWorker(db_params, args.batch, args.n)
    worker.run()