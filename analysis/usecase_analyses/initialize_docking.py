#%%

import psycopg2
from psycopg2.extras import execute_values
import os
import sys
sys.path.append('../../')
from db_utils import db_connection
from tqdm import tqdm

def create_docking_table():
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Create the docking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS docking (
                    id SERIAL PRIMARY KEY,
                    receptor BYTEA NOT NULL,
                    ligand BYTEA NOT NULL,
                    pocket BYTEA,
                    title TEXT,
                    ligand_size INTEGER NOT NULL,
                    docking_status INTEGER NOT NULL DEFAULT 0,
                    docking_output BYTEA,
                    best_score DOUBLE PRECISION,
                    best_pose BYTEA,
                    error_log TEXT,
                    receptor_format TEXT,
                    ligand_format TEXT,
                    pocket_format TEXT
                )
            """)
            
            conn.commit()
            print("Successfully created docking table")

        except Exception as e:
            conn.rollback()
            print(f"Error creating table: {str(e)}")
            raise

def read_sdf_molecules(sdf_path):
    """Read molecules from SDF file and yield (mol_block, title, num_atoms) one at a time"""
    current_mol = []
    title = None
    num_atoms = 0
    
    # First count total molecules for progress bar
    total_mols = 0
    with open(sdf_path, 'r') as f:
        for line in f:
            if line.strip() == "$$$$":
                total_mols += 1
    
    with open(sdf_path, 'r') as f:
        pbar = tqdm(total=total_mols, desc="Reading molecules")
        for line in f:
            # line = line.rstrip()
            current_mol.append(line)
            
            # First line is the title
            if len(current_mol) == 1:
                title = line.strip()
            
            # Third line contains number of atoms and bonds
            elif len(current_mol) == 4:
                try:
                    num_atoms = int(line.split()[0])
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse number of atoms from line: {line}")
                    num_atoms = 0
            
            # Check for molecule end marker
            if line.startswith("$$$$"):
                if current_mol and title and num_atoms > 0:
                    yield ''.join(current_mol), title, num_atoms
                current_mol = []
                title = None
                num_atoms = 0
                pbar.update(1)
        pbar.close()

def upload_structures(protein_path, ligand_path, pocket_path, title, batch_size=10):
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Read the files
            with open(protein_path, 'rb') as f:
                receptor_data = f.read()
            with open(pocket_path, 'rb') as f:
                pocket_data = f.read()

            total_molecules = 0
            data_to_insert = []
            
            # Process molecules one at a time and batch insert
            for mol_block, mol_name, ligand_size in read_sdf_molecules(ligand_path):
                full_title = title + '_' + mol_name
                
                # Convert ligand data to bytes
                ligand_data = mol_block.encode('utf-8')
                
                data_to_insert.append((
                    receptor_data,  # already bytes from rb mode
                    ligand_data,    # converted to bytes
                    pocket_data,    # already bytes from rb mode
                    full_title,
                    ligand_size,
                    0,  # docking_status
                    'pdb',  # receptor_format
                    'sdf',  # ligand_format
                    'mol2'  # pocket_format
                ))

                # When batch size is reached, insert and clear
                if len(data_to_insert) >= batch_size:
                    insert_query = """
                        INSERT INTO docking (
                            receptor, ligand, pocket, title, ligand_size, 
                            docking_status, receptor_format, ligand_format, pocket_format
                        )
                        SELECT * FROM (VALUES %s) AS v(
                            receptor, ligand, pocket, title, ligand_size,
                            docking_status, receptor_format, ligand_format, pocket_format
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM docking d WHERE d.title = v.title
                        )
                    """
                    execute_values(cursor, insert_query, data_to_insert)
                    conn.commit()
                    
                    total_molecules += len(data_to_insert)
                    tqdm.write(f"Uploaded batch of {len(data_to_insert)} molecules. Total: {total_molecules}")
                    data_to_insert = []

            # Insert any remaining molecules
            if data_to_insert:
                insert_query = """
                    INSERT INTO docking (
                        receptor, ligand, pocket, title, ligand_size, 
                        docking_status, receptor_format, ligand_format, pocket_format
                    )
                    SELECT * FROM (VALUES %s) AS v(
                        receptor, ligand, pocket, title, ligand_size,
                        docking_status, receptor_format, ligand_format, pocket_format
                    )
                    WHERE NOT EXISTS (
                        SELECT 1 FROM docking d WHERE d.title = v.title
                    )
                """
                execute_values(cursor, insert_query, data_to_insert)
                conn.commit()
                
                total_molecules += len(data_to_insert)
                tqdm.write(f"Uploaded final batch of {len(data_to_insert)} molecules. Total: {total_molecules}")

            if total_molecules == 0:
                tqdm.write("No molecules were uploaded")
            else:
                tqdm.write(f"Successfully uploaded {total_molecules} molecules to the docking table")

        except Exception as e:
            conn.rollback()
            tqdm.write(f"Error uploading structures: {str(e)}")
            raise

def get_sdf_of_first_row():
    """Retrieve and return the SDF content from the first row of the docking table"""
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Get the first row's ligand data
            cursor.execute("""
                SELECT ligand, title, ligand_format
                FROM docking 
                ORDER BY id 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result:
                ligand_data, title, ligand_format = result
                # Convert bytes back to string
                sdf_content = ligand_data.tobytes().decode('utf-8')
                print(f"\nRetrieved SDF for molecule: {title}")
                print(f"Format: {ligand_format}")
                print("\nSDF Content:")
                # Split the content by newlines and print each line
                for line in sdf_content.split('\n'):
                    print(line)
                return sdf_content
            else:
                print("No data found in the docking table")
                return None
                
        except Exception as e:
            print(f"Error retrieving SDF data: {str(e)}")
            return None

def get_completed_docking_output():
    """Retrieve and display docking output for entries with docking_status=2"""
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Get docking output for completed dockings
            cursor.execute("""
                SELECT docking_output, title, best_score 
                FROM docking 
                WHERE docking_status = 2
                ORDER BY id
            """)
            
            results = cursor.fetchall()
            if results:
                print(f"\nFound {len(results)} completed docking results:")
                for result in results:
                    docking_output, title, best_score = result
                    if docking_output:
                        # Convert bytes back to string
                        output_content = docking_output.tobytes().decode('utf-8')
                        print(f"\nDocking result for molecule: {title}")
                        print(f"Best score: {best_score}")
                        print("\nDocking output:")
                        # Split the content by newlines and print each line
                        for line in output_content.split('\n'):
                            print(line)
                        print("-" * 80)  # Separator between results
                    else:
                        print(f"\nNo docking output found for {title}")
                return results
            else:
                print("No completed docking results found (docking_status = 2)")
                return None
                
        except Exception as e:
            print(f"Error retrieving docking output: {str(e)}")
            return None

def get_docking_scores(target_name):
    """Retrieve docking scores for a specific target"""
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Get docking scores for the specified target
            cursor.execute("""
                SELECT title, best_score, docking_status
                FROM docking 
                WHERE title LIKE %s
                ORDER BY best_score ASC
            """, (f"{target_name}%",))
            
            results = cursor.fetchall()
            if results:
                print(f"\nDocking scores for target {target_name}:")
                print(f"Found {len(results)} entries")
                print("\n{:<30} {:<15} {:<15}".format("Title", "Score", "Status"))
                print("-" * 60)
                
                for title, score, status in results:
                    status_text = "Completed" if status == 2 else "Pending" if status == 0 else "Failed"
                    print("{:<30} {:<15.2f} {:<15}".format(
                        title[:30],  # Truncate long titles
                        score if score is not None else float('inf'),
                        status_text
                    ))
                return results
            else:
                print(f"No docking results found for target {target_name}")
                return None
                
        except Exception as e:
            print(f"Error retrieving docking scores: {str(e)}")
            return None

def get_file_format(file_path):
    """Get file format based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    format_map = {
        '.pdb': 'pdb',
        '.sdf': 'sdf',
        '.mol2': 'mol2',
        '.pdbqt': 'pdbqt'
    }
    return format_map.get(ext, 'unknown')

def upload_native_structures(target_path, ligand_path, title):
    """Upload native structures (protein, ligand, pocket) for a given target"""
    print(f"\nUploading native structures for target {target_path}")
    print(f"Protein: {target_path}")
    print(f"Ligand: {ligand_path}")
    print(f"Pocket: {ligand_path}")
    
    try:
        # Check if files exist
        for file_path in [target_path, ligand_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Get file formats
        receptor_format = get_file_format(target_path)
        ligand_format = get_file_format(ligand_path)
        
        if receptor_format == 'unknown':
            raise ValueError(f"Unsupported receptor file format: {target_path}")
        if ligand_format == 'unknown':
            raise ValueError(f"Unsupported ligand file format: {ligand_path}")
        
        with db_connection() as conn:
            cursor = conn.cursor()
            try:
                # Read the files
                with open(target_path, 'rb') as f:
                    receptor_data = f.read()
                with open(ligand_path, 'rb') as f:
                    ligand_data = f.read()
                
                # Insert the native structure
                insert_query = """
                    INSERT INTO docking (
                        receptor, ligand, pocket, title, ligand_size, 
                        docking_status, receptor_format, ligand_format, pocket_format
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Get ligand size based on file format
                ligand_size = 0
                if ligand_format == 'sdf':
                    with open(ligand_path, 'r') as f:
                        for i, line in enumerate(f):
                            if i == 3:  # Fourth line contains number of atoms
                                try:
                                    ligand_size = int(line.split()[0])
                                except (ValueError, IndexError):
                                    print(f"Warning: Could not parse number of atoms from line: {line}")
                                break
                elif ligand_format == 'mol2':
                    with open(ligand_path, 'r') as f:
                        for line in f:
                            if line.startswith('@<TRIPOS>ATOM'):
                                break
                        # Count atoms until @<TRIPOS>BOND
                        for line in f:
                            if line.startswith('@<TRIPOS>BOND'):
                                break
                            if line.strip() and not line.startswith('@'):
                                ligand_size += 1
                
                cursor.execute(insert_query, (
                    receptor_data,
                    ligand_data,
                    ligand_data,  # Using ligand file as pocket
                    title,
                    ligand_size,
                    0,  # docking_status
                    receptor_format,  # receptor_format
                    ligand_format,  # ligand_format
                    ligand_format  # pocket_format (using same as ligand)
                ))
                
                conn.commit()
                print(f"Successfully uploaded native structure for {title}")
                print(f"Receptor format: {receptor_format}")
                print(f"Ligand format: {ligand_format}")
                print(f"Number of atoms: {ligand_size}")
                
            except Exception as e:
                conn.rollback()
                print(f"Error uploading structures: {str(e)}")
                raise
                
    except Exception as e:
        print(f"Error uploading native structures: {str(e)}")
        raise

#%%
create_docking_table()

# protein_path = '7ckz.pdb'
# ligand_path = '7ckz_generation_2d.sdf'
# pocket_path = '7ckz_ligand.mol2'
# upload_structures(protein_path, ligand_path, pocket_path, '7ckz', batch_size=10)

protein_path = '7e2y.pdb'
ligand_path = '7e2y_generation_2d.sdf'
pocket_path = '7e2y_ligand.mol2'
upload_structures(protein_path, ligand_path, pocket_path, '7e2y')

#%%

upload_native_structures('7ckz.pdb', '7ckz_ligand.mol2', '7ckz_native')
upload_native_structures('7e2y.pdb', '7e2y_ligand.mol2', '7e2y_native')

#%%
print("\nTesting SDF retrieval from first row:")
get_sdf_of_first_row()

print("\nChecking completed docking results:")
get_completed_docking_output()

print("\nGetting docking scores:")
get_docking_scores('7e2y')

#%%
