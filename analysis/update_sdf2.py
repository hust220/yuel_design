import psycopg2
from tqdm import tqdm
import sys
sys.path.append('../')
from db_utils import db_connection

BATCH_SIZE = 1000  # Number of molecules to process in each batch

def add_sdf2_column_if_not_exists():
    """Add sdf2 column to molecules table if it doesn't exist"""
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='molecules' AND column_name='sdf2'
                """)
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE molecules ADD COLUMN sdf2 bytea")
                    conn.commit()
                    print("Added sdf2 column to molecules table")
        except Exception as e:
            conn.rollback()
            raise

def parse_sdf_file(input_filename):
    """
    Parse SDF file and yield tuples of (molecule_id, sdf_record)
    """
    with open(input_filename, 'r', encoding='utf-8') as f:
        current_record = []

        for line in f:
            current_record.append(line)
                            
            # Check for end of record
            if line.strip() == '$$$$':
                yield ''.join(current_record).encode('utf-8')
                current_record = []

def get_ids_from_sdf(input_filename):
    """Get all molecule IDs from SDF file"""
    ids = []
    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('> <_ID>'):
                # Next line contains the ID
                current_id = next(f).strip()
                ids.append(int(current_id))
    return ids

def update_sdf2_in_database(input_filename1, input_filename2):
    """Update sdf2 column from processed SDF file"""
    add_sdf2_column_if_not_exists()

    ids = get_ids_from_sdf(input_filename1)
    
    # First count total records in SDF file for progress bar
    print("Counting records in SDF file...")
    total_records = 0
    with open(input_filename2, 'r', encoding='utf-8') as f:
        total_records = sum(1 for line in f if line.strip() == '$$$$')
    
    imol = 0
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                # Process records in batches
                batch = []
                processed = 0
                
                for sdf_data in tqdm(parse_sdf_file(input_filename2), 
                                                 total=total_records, 
                                                 desc="Updating database"):
                    batch.append((sdf_data, ids[imol]))
                    imol += 1
                    
                    if len(batch) >= BATCH_SIZE:
                        # Update batch
                        cursor.executemany("""
                            UPDATE molecules 
                            SET sdf2 = %s 
                            WHERE id = %s
                        """, batch)
                        conn.commit()
                        processed += len(batch)
                        batch = []
                
                # Update remaining records
                if batch:
                    cursor.executemany("""
                        UPDATE molecules 
                        SET sdf2 = %s 
                        WHERE id = %s
                    """, batch)
                    conn.commit()
                    processed += len(batch)
                
                print(f"\nSuccessfully updated {processed} records")
                
        except Exception as e:
            conn.rollback()
            print(f"Database error: {str(e)}", file=sys.stderr)
            raise

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_sdf_file1> <input_sdf_file2>")
        sys.exit(1)
    
    input_filename1 = sys.argv[1]
    input_filename2 = sys.argv[2]
    update_sdf2_in_database(input_filename1, input_filename2)

def test():
    # select on sdf2 from molecules where id = 1539
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT sdf2 FROM molecules WHERE id = 1539")
            sdf = cursor.fetchone()[0].tobytes().decode('utf-8')
            print(sdf)

if __name__ == "__main__":
    # test()
    main()