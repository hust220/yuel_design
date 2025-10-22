#%%

import os
from tqdm import tqdm
import sys
sys.path.append('../')
from db_utils import db_connection
import psycopg2.extras
from rdkit import Chem
from rdkit import RDLogger
import matplotlib.pyplot as plt

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def create_gpr75_table():
    """Create a new table for GPR75 molecules"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gpr75_molecules (
                    id SERIAL PRIMARY KEY,
                    molecule_name TEXT NOT NULL,
                    sdf BYTEA NOT NULL,
                    size INTEGER NOT NULL CHECK (size > 0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()

def add_validity_column():
    """Add validity column to the gpr75_molecules table"""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                ALTER TABLE gpr75_molecules 
                ADD COLUMN IF NOT EXISTS validity BOOLEAN
            """)
        conn.commit()

def mol_size(block):
    """Extract molecule size from SDF block"""
    iline = 0
    for line in block.split('\n'):
        if iline == 3:
            ls = line.strip().split()
            if len(ls) == 11:
                return int(ls[0])
        iline += 1
    return None

def is_valid(mol):
    """Check if a molecule is valid using RDKit's SanitizeMol"""
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

def scan_sdf(sdf_file):
    """Scan SDF file and yield molecule blocks with metadata"""
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
            yield block, name, size
            block = ""
            iline = 0

def insert_batch(conn, batch):
    """Insert a batch of records into the database."""
    with conn.cursor() as cursor:
        psycopg2.extras.execute_batch(cursor, """
            INSERT INTO gpr75_molecules (molecule_name, size, sdf)
            VALUES (%s, %s, %s)
        """, batch)
    conn.commit()

def load_gpr75_molecules(sdf_file):
    """Load molecules from SDF file and store them in the database"""
    if not os.path.exists(sdf_file):
        raise FileNotFoundError(f"SDF file not found: {sdf_file}")
    
    batch = []
    BATCH_SIZE = 1000
    
    with db_connection() as conn:
        try:
            for sdf_block, name, size in tqdm(scan_sdf(sdf_file), desc="Loading molecules"):
                batch.append((
                    name,
                    size,
                    sdf_block.encode('utf-8')
                ))
                
                if len(batch) >= BATCH_SIZE:
                    insert_batch(conn, batch)
                    batch = []

            if batch:
                insert_batch(conn, batch)
                
        except Exception as e:
            print(f"Error loading molecules: {str(e)}")
            raise

def calculate_validity_for_existing_molecules():
    """Calculate validity for all existing molecules in the GPR75 table"""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get total count of molecules
        cursor.execute("SELECT COUNT(*) FROM gpr75_molecules")
        total_molecules = cursor.fetchone()[0]
        
        # Process in batches
        batch_size = 100
        for offset in tqdm(range(0, total_molecules, batch_size), desc="Processing batches"):
            # Get batch of molecules
            cursor.execute("""
                SELECT id, sdf 
                FROM gpr75_molecules 
                ORDER BY id 
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            rows = cursor.fetchall()
            
            # Process each molecule in the batch
            for mol_id, sdf_bytes in tqdm(rows, desc=f"Analyzing molecules in batch {offset//batch_size + 1}", leave=False):
                # Convert SDF bytes to molecule and check validity
                mol = Chem.MolFromMolBlock(sdf_bytes.tobytes().decode('utf-8'), sanitize=False)
                validity = is_valid(mol) if mol is not None else False
                
                # Update validity in database
                cursor.execute("""
                    UPDATE gpr75_molecules 
                    SET validity = %s
                    WHERE id = %s
                """, (validity, mol_id))
            
            # Commit after each batch
            conn.commit()

def save_valid_molecules_to_file(output_file):
    """Save all valid molecules to a single file.
    
    Args:
        output_file (str): Path to the output file where valid molecules will be saved
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get total count of valid molecules
        cursor.execute("SELECT COUNT(*) FROM gpr75_molecules WHERE validity = true")
        total_molecules = cursor.fetchone()[0]
        print(f"Found {total_molecules} valid molecules")
        
        if total_molecules == 0:
            print("No valid molecules found in database")
            return
        
        # Process in batches
        batch_size = 100
        with open(output_file, 'w') as f:
            for offset in tqdm(range(0, total_molecules, batch_size), desc="Saving molecules"):
                # Get batch of molecules
                cursor.execute("""
                    SELECT sdf 
                    FROM gpr75_molecules 
                    WHERE validity = true
                    ORDER BY id 
                    LIMIT %s OFFSET %s
                """, (batch_size, offset))
                rows = cursor.fetchall()
                
                # Write each molecule's SDF block to file
                for sdf_bytes, in rows:
                    sdf_str = sdf_bytes.tobytes().decode('utf-8')
                    f.write(sdf_str)
        
        print(f"Successfully saved {total_molecules} valid molecules to {output_file}")

def plot_validity_vs_size():
    """Plot the relationship between molecule size and validity rate."""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get size and validity data
        cursor.execute("""
            SELECT size, 
                   COUNT(*) as total_count,
                   SUM(CASE WHEN validity = true THEN 1 ELSE 0 END) as valid_count
            FROM gpr75_molecules
            GROUP BY size
            ORDER BY size
        """)
        rows = cursor.fetchall()
        
        if not rows:
            print("No data found in database")
            return
        
        # Prepare data for plotting
        sizes = []
        validity_rates = []
        total_counts = []
        
        for size, total, valid in rows:
            sizes.append(size)
            validity_rates.append(100 * valid / total)  # Convert to percentage
            total_counts.append(total)
        
        # Create figure
        plt.figure(figsize=(6, 4))
        
        # Plot validity rate vs size
        scatter = plt.scatter(sizes, validity_rates, c=total_counts, cmap='viridis', 
                          alpha=0.6, s=50)
        plt.xlabel('Molecule Size (Number of Atoms)')
        plt.ylabel('Validity Rate (%)')
        plt.title('Molecule Validity Rate vs Size')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar to show count scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Number of Molecules')
        
        # Calculate and display statistics
        total_molecules = sum(total_counts)
        total_valid = sum(v * t / 100 for v, t in zip(validity_rates, total_counts))
        overall_validity_rate = 100 * total_valid / total_molecules
        
        stats_text = f'Total Molecules: {total_molecules:,}\n'
        stats_text += f'Overall Validity Rate: {overall_validity_rate:.1f}%'
        plt.text(0.02, 0.98, stats_text, 
               transform=plt.gca().transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig('GPR75_validity_vs_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved as GPR75_validity_vs_size.png")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total number of molecules: {total_molecules:,}")
        print(f"Overall validity rate: {overall_validity_rate:.1f}%")
        print(f"Size range: {min(sizes)} to {max(sizes)} atoms")
        
        # Find size with highest validity rate
        max_validity_idx = validity_rates.index(max(validity_rates))
        print(f"\nHighest validity rate: {validity_rates[max_validity_idx]:.1f}% "
              f"for molecules with {sizes[max_validity_idx]} atoms "
              f"(sample size: {total_counts[max_validity_idx]})")

if __name__ == '__main__':
    # sdf_file = 'GPR75_generation_2d.sdf'
    
    # print("Creating GPR75 molecules table...")
    # create_gpr75_table()
    
    # print(f"Loading molecules from {sdf_file}...")
    # load_gpr75_molecules(sdf_file)
    
    # print("Adding validity column...")
    # add_validity_column()
    
    # print("Calculating validity for all molecules...")
    # calculate_validity_for_existing_molecules()

    # Save valid molecules to file
    # save_valid_molecules_to_file('GPR75_molecules_valid.sdf')

    # Plot validity rate vs size
    # plot_validity_vs_size()
    
    # print("Done!")


# %%
