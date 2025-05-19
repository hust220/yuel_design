import psycopg2
from psycopg2 import sql
from tqdm import tqdm
import sys
from io import BytesIO
sys.path.append('../')
from db_utils import db_connection

BATCH_SIZE = 1000  # Number of molecules to process in each batch

def extract_sdf_to_file(output_filename):
    """Extract SDF data from molecules table in batches and write to a single SDF file"""
    with db_connection() as conn:
        try:
            with conn.cursor() as cursor:
                # Get total count for progress bar
                cursor.execute("SELECT COUNT(*) FROM molecules")
                total_molecules = cursor.fetchone()[0]
                
                # Open output file
                with open(output_filename, 'wb') as outfile:
                    # Process in batches
                    offset = 0
                    processed = 0
                    
                    with tqdm(total=total_molecules, desc="Writing SDFs") as pbar:
                        while True:
                            # Fetch a batch of molecules
                            cursor.execute("""
                                SELECT id, sdf 
                                FROM molecules 
                                ORDER BY id
                                LIMIT %s OFFSET %s
                            """, (BATCH_SIZE, offset))
                            
                            batch = cursor.fetchall()
                            if not batch:
                                break
                            
                            for molecule_id, sdf_data in batch:
                                try:
                                    # Convert bytea to bytes
                                    sdf_bytes = sdf_data.tobytes()
                                    
                                    # Split into individual SDF records (assuming each record ends with $$$$)
                                    sdf_str = sdf_bytes.decode('utf-8')
                                    records = sdf_str.split('$$$$\n')
                                    
                                    for record in records:
                                        if not record.strip():
                                            continue
                                            
                                        # Add _ID property to the record
                                        modified_record = record
                                        if '> <' not in modified_record:  # If no properties exist
                                            modified_record += '\n> <_ID>\n'
                                        else:
                                            # Insert _ID property before the first existing property
                                            prop_start = modified_record.find('> <')
                                            modified_record = modified_record[:prop_start] + f'> <_ID>\n{molecule_id}\n' + modified_record[prop_start:] + '$$$$\n'
                                        
                                        # Write to output file
                                        outfile.write(modified_record.encode('utf-8'))
                                        
                                except Exception as e:
                                    print(f"Error processing molecule ID {molecule_id}: {str(e)}", file=sys.stderr)
                                    continue
                            
                            offset += BATCH_SIZE
                            processed += len(batch)
                            pbar.update(len(batch))
                            
        except Exception as e:
            conn.rollback()
            print(f"Database error: {str(e)}", file=sys.stderr)
            raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <output_sdf_file>")
        sys.exit(1)
    
    output_filename = sys.argv[1]
    extract_sdf_to_file(output_filename)
    print(f"\nSuccessfully wrote SDF data to {output_filename}")

if __name__ == "__main__":
    main()