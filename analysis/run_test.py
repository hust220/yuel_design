import os
import sys
import time
import random
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection
from src.console import section, info, success, warn, error
from yuel_design import run_design, pick_device


OUTPUT_DIR = 'test_split_designs'
CHECKPOINT = None
DEVICE = 'auto'
N_POCKETS = 100
MIN_SIZE = 15
MAX_SIZE = 35
SEED = None
RUN_ID = 1


def get_test_pockets(limit=100):
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT id, pdb
                FROM moad_pockets 
                WHERE split = 'test'
                ORDER BY id
                LIMIT %s
                """,
                (limit,),
            )
            return c.fetchall()


def create_results_table():
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS moad_test (
                    id SERIAL PRIMARY KEY,
                    pocket_id TEXT NOT NULL,
                    ligand_size INTEGER NOT NULL,
                    run_id INTEGER NOT NULL,
                    seed INTEGER,
                    output_pdb_path TEXT,
                    trajectory_path TEXT,
                    log_path TEXT,
                    is_valid BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    n_attempts INTEGER NOT NULL,
                    n_failures_ring_size INTEGER DEFAULT 0,
                    n_failures_intact INTEGER DEFAULT 0,
                    n_failures_kekulization INTEGER DEFAULT 0,
                    duration_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pocket_id, ligand_size, run_id)
                )
            """)
            conn.commit()
            success("Created/verified moad_test table")


def run_design_for_pocket(pocket_id, pdb_bytes, ligand_size, output_base_dir, device, checkpoint=None, seed=None):
    output_dir = os.path.join(output_base_dir, pocket_id, f"size_{ligand_size}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_pdb = os.path.join(output_dir, "predicted_structure.pdb")
    trajectory_path = os.path.join(output_dir, "trajectory.pdb")
    log_path = os.path.join(output_dir, "validation.log")
    
    start_time = time.time()
    
    try:
        result = run_design(
            pocket_structure=pdb_bytes,
            output_pdb_path=output_pdb,
            ligand_size=ligand_size,
            checkpoint=checkpoint,
            device=device,
            seed=seed,
            save_trajectory=trajectory_path,
            log_path=log_path,
            max_attempts=20,
            verbose=False
        )
        
        duration = time.time() - start_time
        
        return {
            'is_valid': result['is_valid'],
            'failure_reason': result['failure_reason'] if not result['is_valid'] else None,
            'n_attempts': result['n_attempts'],
            'n_failures_ring_size': result['n_failures_ring_size'],
            'n_failures_intact': result['n_failures_intact'],
            'n_failures_kekulization': result['n_failures_kekulization'],
            'duration_seconds': duration,
            'output_pdb_path': output_pdb if os.path.exists(output_pdb) else None,
            'trajectory_path': trajectory_path if os.path.exists(trajectory_path) else None,
            'log_path': log_path if os.path.exists(log_path) else None,
        }
        
    except Exception as e:
        duration = time.time() - start_time
        error(f"Error running design: {e}")
        
        return {
            'is_valid': False,
            'failure_reason': str(e),
            'n_attempts': 1,
            'n_failures_ring_size': 0,
            'n_failures_intact': 0,
            'n_failures_kekulization': 0,
            'duration_seconds': duration,
            'output_pdb_path': output_pdb if os.path.exists(output_pdb) else None,
            'trajectory_path': trajectory_path if os.path.exists(trajectory_path) else None,
            'log_path': log_path if os.path.exists(log_path) else None,
        }


def save_result_to_db(pocket_id, ligand_size, seed, result):
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO moad_test 
                (pocket_id, ligand_size, run_id, seed, output_pdb_path, trajectory_path, log_path,
                 is_valid, failure_reason, n_attempts, n_failures_ring_size, 
                 n_failures_intact, n_failures_kekulization, duration_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pocket_id, ligand_size, run_id) 
                DO UPDATE SET
                    seed = EXCLUDED.seed,
                    output_pdb_path = EXCLUDED.output_pdb_path,
                    trajectory_path = EXCLUDED.trajectory_path,
                    log_path = EXCLUDED.log_path,
                    is_valid = EXCLUDED.is_valid,
                    failure_reason = EXCLUDED.failure_reason,
                    n_attempts = EXCLUDED.n_attempts,
                    n_failures_ring_size = EXCLUDED.n_failures_ring_size,
                    n_failures_intact = EXCLUDED.n_failures_intact,
                    n_failures_kekulization = EXCLUDED.n_failures_kekulization,
                    duration_seconds = EXCLUDED.duration_seconds
            """, (
                pocket_id,
                ligand_size,
                RUN_ID,
                seed,
                result['output_pdb_path'],
                result['trajectory_path'],
                result['log_path'],
                result['is_valid'],
                result['failure_reason'],
                result['n_attempts'],
                result['n_failures_ring_size'],
                result['n_failures_intact'],
                result['n_failures_kekulization'],
                result['duration_seconds'],
            ))
            conn.commit()


def main():
    global RUN_ID
    
    if len(sys.argv) > 1:
        RUN_ID = int(sys.argv[1])
    
    random.seed(RUN_ID)
    np.random.seed(RUN_ID)
    
    section("Test Split Design Runner")
    
    info(f"Run ID: {RUN_ID}")
    if CHECKPOINT is not None:
        info(f"Checkpoint: {CHECKPOINT}")
    device = pick_device(DEVICE)
    info(f"Using device: {device}")
    
    create_results_table()
    
    pockets = get_test_pockets(limit=N_POCKETS)
    info(f"Found {len(pockets)} test pockets")
    
    if len(pockets) == 0:
        error("No test pockets found!")
        return
    
    output_base_dir = OUTPUT_DIR
    os.makedirs(output_base_dir, exist_ok=True)
    
    ligand_sizes = list(range(MIN_SIZE, MAX_SIZE + 1))
    info(f"Will run {len(ligand_sizes)} designs per pocket (sizes {MIN_SIZE}-{MAX_SIZE})")
    
    total_designs = len(pockets) * len(ligand_sizes)
    info(f"Total designs to run: {total_designs}")
    
    design_count = 0
    
    for pocket_id, pdb_bytes in pockets:
        pocket_id_str = str(pocket_id)
        info(f"Processing pocket {pocket_id_str} ({design_count // len(ligand_sizes) + 1}/{len(pockets)})")
        
        for ligand_size in ligand_sizes:
            design_count += 1
            seed = np.random.randint(0, 2**31)
            
            info(f"  Design {design_count}/{total_designs}: pocket={pocket_id_str}, size={ligand_size}, seed={seed}")
            
            result = run_design_for_pocket(
                pocket_id=pocket_id_str,
                pdb_bytes=pdb_bytes,
                ligand_size=ligand_size,
                output_base_dir=output_base_dir,
                device=device,
                checkpoint=CHECKPOINT,
                seed=seed,
            )
            
            save_result_to_db(pocket_id_str, ligand_size, seed, result)
            
            if result['is_valid']:
                success(f"    Valid design generated in {result['duration_seconds']:.2f}s")
            else:
                warn(f"    Failed: {result['failure_reason']}")
    
    success(f"Completed all {total_designs} designs!")


if __name__ == '__main__':
    main()

