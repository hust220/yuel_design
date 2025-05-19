import psycopg2
from psycopg2 import sql
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
from io import BytesIO
from rdkit.RDLogger import DisableLog
import sys
sys.path.append('../')
from db_utils import db_connection

DisableLog('rdApp.*')

BATCH_SIZE = 10  # Number of molecules to process in each batch
NUM_PROCESSES = 16  # Number of processes to use for parallel processing

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs

def get_fingerprint(mol, fp_type="MACCS", ignore_atom_types=False, ignore_bond_types=False, ignore_bonds=False):
    if ignore_bonds:
        atoms = [a.GetSymbol() if not ignore_atom_types else 'C' for a in mol.GetAtoms()]
        smiles = '.'.join(atoms)
        mol = Chem.MolFromSmiles(smiles)

    if ignore_atom_types or ignore_bond_types:
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            if ignore_atom_types:
                atom.SetAtomicNum(6)
        for bond in mol.GetBonds():
            if ignore_bond_types:
                bond.SetBondType(Chem.BondType.SINGLE)

    # 选择指纹类型
    if fp_type == "MACCS":
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "Topological":
        return Chem.RDKFingerprint(mol)
    elif fp_type == "AtomPair":
        return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
    elif fp_type == "Torsion":
        return AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")

def calculate_similarity(mol1, mol2, fp_type="MACCS", ignore_atom_types=False, ignore_bond_types=False, ignore_bonds=False):
    fp1 = get_fingerprint(mol1, fp_type=fp_type, ignore_atom_types=ignore_atom_types, ignore_bond_types=ignore_bond_types, ignore_bonds=ignore_bonds)
    fp2 = get_fingerprint(mol2, fp_type=fp_type, ignore_atom_types=ignore_atom_types, ignore_bond_types=ignore_bond_types, ignore_bonds=ignore_bonds)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calculate_similarities(mol1, mol2):
    sim1 = calculate_similarity(mol1, mol2)
    sim2 = calculate_similarity(mol1, mol2, ignore_atom_types=True)
    sim3 = calculate_similarity(mol1, mol2, ignore_bond_types=True)
    sim4 = calculate_similarity(mol1, mol2, ignore_atom_types=True, ignore_bond_types=True)
    sim5 = calculate_similarity(mol1, mol2, ignore_bonds=True)
    return sim1, sim2, sim3, sim4, sim5

def add_similarity_column():
    """Add similarity column to molecules table if it doesn't exist"""
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='molecules' AND column_name='similarity'
                """)
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE molecules ADD COLUMN similarity text")
                    conn.commit()
                    print("Added similarity column to molecules table")
        except Exception as e:
            conn.rollback()
            raise

def get_molecule_ids_to_process():
    """Get all molecule IDs that need processing (where similarity is NULL)"""
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT m.id
                    FROM molecules m
                    JOIN ligands l ON m.ligand_name = l.name
                    WHERE m.similarity IS NULL
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            conn.rollback()
            raise

def get_ligands_for_names(ligand_names):
    """Load ligands for specific names"""
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT name, mol FROM ligands WHERE name = ANY(%s)
                """, (ligand_names,))
                return {name: mol.tobytes().decode('utf-8') for name, mol in cursor.fetchall()}
        except Exception as e:
            conn.rollback()
            raise

def get_molecules_batch(molecule_ids):
    """Get a batch of molecule data"""
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, sdf, ligand_name, size 
                    FROM molecules 
                    WHERE id = ANY(%s)
                """, (molecule_ids,))
                return cursor.fetchall()
        except Exception as e:
            conn.rollback()
            raise

def process_batch(batch):
    try:
        batch_data = get_molecules_batch(batch)
    except Exception as e:
        print(f"Error getting molecules batch: {str(e)}", file=sys.stderr)
        return [(molecule_id, False) for molecule_id in batch]

    # First collect all unique ligand names needed for this batch
    ligand_names = list(set(row[2] for row in batch_data))
    
    # Load only the ligands needed for this batch
    ligand_data = get_ligands_for_names(ligand_names)
    
    results = []
    processed_data = []
    
    for molecule_id, sdf_data, ligand_name, size in batch_data:
        try:
            # Get corresponding ligand mol
            ligand_mol_data = ligand_data.get(ligand_name)
            if not ligand_mol_data:
                print(f"Ligand {ligand_name} not found")
                results.append((molecule_id, False))
                continue
            
            # Parse ligand molecule
            try:
                # load from mol block
                ligand_mol = Chem.MolFromMolBlock(ligand_mol_data)
                if not ligand_mol:
                    print(f"Ligand {ligand_name} converted to None")
                    results.append((molecule_id, False))
                    continue
            except:
                print(f"Ligand {ligand_name} conversion failed")
                results.append((molecule_id, False))
                continue
            
            # Parse SDF (might contain multiple molecules)
            try:
                sdf_data = sdf_data.tobytes()
                sdf_data = BytesIO(sdf_data)
                sdf_mols = [mol for mol in Chem.ForwardSDMolSupplier(sdf_data, sanitize=False, strictParsing=False) if mol is not None]
            except:
                print(f"SDF {sdf_data} conversion failed")
                results.append((molecule_id, False))
                continue
            
            if not sdf_mols:
                print(f"SDF {sdf_data} conversion returned no molecules")
                results.append((molecule_id, False))
                continue
            
            # Calculate similarities
            similarities = []
            for mol in sdf_mols:
                try:
                    sims = calculate_similarities(mol, ligand_mol)
                    if sims is not None:
                        similarities.append(sims)
                except:
                    print(f"Similarity calculation failed for molecule {molecule_id}")
                    continue
            
            if not similarities:
                print(f"Similarity calculation returned no values for molecule {molecule_id}")
                results.append((molecule_id, False))
                continue
            # print(similarities)
            # Store result for batch update
            similarity_value = '\n'.join([' '.join(map(str, sims)) for sims in similarities])
            # print(similarity_value)
            processed_data.append((similarity_value, molecule_id))
            results.append((molecule_id, True))
            
        except Exception as e:
            print(f"Error processing molecule {molecule_id}: {str(e)}", file=sys.stderr)
            # raise e
            results.append((molecule_id, False))
    
    # Perform batch update if we have results
    if processed_data:
        with db_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.executemany("""
                        UPDATE molecules 
                        SET similarity = %s 
                        WHERE id = %s
                    """, processed_data)
                    conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Batch update failed: {str(e)}", file=sys.stderr)
    
    return results

def process_molecules_parallel(molecule_ids, num_processes=None):
    """Process molecules in parallel with batch processing"""
    if num_processes is None:
        num_processes = cpu_count()
    
    # Split molecule_ids into batches
    batches = [molecule_ids[i:i + BATCH_SIZE] for i in range(0, len(molecule_ids), BATCH_SIZE)]
    
    total_processed = 0
    total_successful = 0
    
    with Pool(processes=num_processes) as pool:
        # Process batches with progress tracking
        for batch in tqdm(batches, desc="Processing batches"):
            # Get all molecule data for this batch
            try:
                batch_results = process_batch(batch)
                
                # Update statistics
                successful = sum(1 for _, success in batch_results if success)
                total_processed += len(batch_results)
                total_successful += successful
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}", file=sys.stderr)
                continue
    
    print(f"\nProcessed {total_processed} molecules, {total_successful} successful")

def main():
    # Add similarity column if it doesn't exist
    add_similarity_column()
    
    # Get all molecule IDs that need processing
    molecule_ids = get_molecule_ids_to_process()
    print(f"Found {len(molecule_ids)} molecules to process")
    
    if not molecule_ids:
        print("No molecules to process")
        return
    
    # Process molecules in parallel with batch processing
    process_molecules_parallel(molecule_ids, NUM_PROCESSES)

if __name__ == "__main__":
    main()
