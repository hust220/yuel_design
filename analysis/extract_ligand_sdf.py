import argparse
import os
import sys
from pathlib import Path

from psycopg2 import sql
from rdkit import Chem, Geometry
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.console import section, info, success, warn, error
from src.db_utils import db_connection, ensure_column_exists
from src.pdb_utils import Structure

DEFAULT_TABLE = "moad_test"
LIGAND_COLUMN = "ligand_sdf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ligand SDF blocks from predicted PDBs and store them in the database."
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Target results table (default: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: all pending).",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Recompute ligand SDFs even for rows where the column is already populated.",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="Only process rows with is_valid = TRUE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and log ligand SDFs without updating the database.",
    )
    return parser.parse_args()


def ensure_schema(table_name: str):
    ensure_column_exists(table_name, LIGAND_COLUMN, "TEXT")


def fetch_rows_iter(table_name: str, limit: int = None, include_existing: bool = False, only_valid: bool = False):
    """
    Generator that yields rows one at a time to avoid loading all into memory.
    Processes rows incrementally to minimize memory footprint.
    """
    query = [
        f"""
        SELECT id, pocket_id, output_pdb_path, ligand_size, is_valid
        FROM {table_name}
        WHERE output_pdb_path IS NOT NULL
        """
    ]
    params = []
    if not include_existing:
        query.append("AND (ligand_sdf IS NULL OR ligand_sdf = '')")
    if only_valid:
        query.append("AND is_valid = TRUE")
    query.append("ORDER BY id")
    if limit is not None:
        query.append("LIMIT %s")
        params.append(limit)

    sql_query = " ".join(query)

    with db_connection() as conn:
        with conn.cursor() as c:
            # Execute query - results are fetched lazily
            c.execute(sql_query, params)
            # Fetch rows one at a time to minimize memory usage
            while True:
                row = c.fetchone()
                if row is None:
                    break
                yield row


def isolate_ligand_mol(pdb_path: str):
    """
    Extract ligand molecule from PDB file using pdb_utils.
    Returns None if no ligand found.
    Memory-efficient: processes and returns only the ligand molecule.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Use pdb_utils to parse PDB file (don't skip HETATM, don't skip water)
    structure = Structure(pdb_path, skip_hetatm=False, skip_water=False)
    
    # Extract ligand atoms (residue name = "LIG")
    ligand_atoms_data = []  # List of (element, coords) tuples
    
    try:
        for model in structure.models:
            for chain in model:
                for residue in chain:
                    # Check if this is a ligand residue
                    # In yuel_design (src/e2efinal/app.py), ligands are saved with:
                    #   - res_name='LIG'
                    #   - record='HETATM'
                    #   - chain_id='B' (typically)
                    # We check res_name='LIG' as the primary identifier for robustness
                    if residue.res_name.strip().upper() == "LIG":
                        for atom in residue:
                            element = atom.element.strip()
                            if not element:
                                # If element is missing, try to infer from atom name
                                # PDB format: element is in columns 76-77, but may be missing
                                # Try to infer from atom name (first 1-2 characters)
                                atom_name = atom.atom_name.strip()
                                if atom_name:
                                    # Common two-character elements
                                    if atom_name.startswith('CL') or atom_name.startswith('Cl'):
                                        element = 'Cl'
                                    elif atom_name.startswith('BR') or atom_name.startswith('Br'):
                                        element = 'Br'
                                    elif atom_name.startswith('MG') or atom_name.startswith('Mg'):
                                        element = 'Mg'
                                    elif atom_name.startswith('ZN') or atom_name.startswith('Zn'):
                                        element = 'Zn'
                                    elif atom_name.startswith('FE') or atom_name.startswith('Fe'):
                                        element = 'Fe'
                                    else:
                                        # Take first character
                                        element = atom_name[0].upper()
                                else:
                                    element = 'C'  # Default fallback
                            coords = atom.get_coord()
                            ligand_atoms_data.append((element, coords))
    finally:
        # Explicitly delete structure to free memory
        del structure
    
    if not ligand_atoms_data:
        return None
    
    # Build RDKit molecule from extracted atoms
    mol = Chem.RWMol()
    
    # Normalize element symbols and create atoms
    def normalize_element(elem):
        """Normalize element symbol to RDKit-compatible format."""
        elem = elem.strip()
        if not elem:
            return 'C'
        # Handle two-character elements
        if len(elem) >= 2:
            elem_upper = elem.upper()
            if elem_upper.startswith('CL'):
                return 'Cl'
            elif elem_upper.startswith('BR'):
                return 'Br'
            elif elem_upper.startswith('MG'):
                return 'Mg'
            elif elem_upper.startswith('ZN'):
                return 'Zn'
            elif elem_upper.startswith('FE'):
                return 'Fe'
            elif elem_upper.startswith('CA'):
                return 'Ca'
            elif elem_upper.startswith('NA'):
                return 'Na'
            elif elem_upper.startswith('AL'):
                return 'Al'
            elif elem_upper.startswith('SI'):
                return 'Si'
            else:
                # Take first character for single-character elements
                return elem[0].upper()
        else:
            return elem.upper()
    
    # Add atoms
    atom_indices = []
    normalized_elements = []
    for element, coords in ligand_atoms_data:
        norm_elem = normalize_element(element)
        normalized_elements.append(norm_elem)
        try:
            atom = Chem.Atom(norm_elem)
        except:
            # If RDKit doesn't recognize the element, default to carbon
            atom = Chem.Atom('C')
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)
    
    # Add conformer with coordinates
    conf = Chem.Conformer(len(ligand_atoms_data))
    for i, (element, coords) in enumerate(ligand_atoms_data):
        conf.SetAtomPosition(i, Geometry.Point3D(float(coords[0]), float(coords[1]), float(coords[2])))
    mol.AddConformer(conf)
    
    # Infer bonds based on distances
    coords_array = np.array([coords for _, coords in ligand_atoms_data])
    n_atoms = len(ligand_atoms_data)
    
    if n_atoms > 1:
        # Compute distance matrix
        dists = np.linalg.norm(coords_array[:, None, :] - coords_array[None, :, :], axis=2)
        
        # Bond length thresholds (in Angstroms)
        bond_lengths = {
            ('C', 'C'): 1.8, ('C', 'N'): 1.7, ('C', 'O'): 1.6, ('C', 'S'): 2.0,
            ('N', 'N'): 1.6, ('N', 'O'): 1.5, ('O', 'O'): 1.5, ('S', 'S'): 2.2,
            ('C', 'F'): 1.5, ('C', 'Cl'): 2.0, ('C', 'Br'): 2.1, ('C', 'I'): 2.3,
            ('N', 'F'): 1.4, ('O', 'F'): 1.4, ('S', 'F'): 1.8,
        }
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                elem_i = normalized_elements[i]
                elem_j = normalized_elements[j]
                
                # Get max bond length for this pair
                pair = tuple(sorted([elem_i, elem_j]))
                max_bond_length = bond_lengths.get(pair, 2.5)
                
                if dists[i, j] < max_bond_length:
                    mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
    
    ligand_mol = mol.GetMol()
    del mol  # Free memory
    
    # Try to sanitize, but don't fail if it doesn't work
    try:
        Chem.SanitizeMol(ligand_mol)
    except Exception:
        warn("Sanitization issues detected; storing raw molecule.")
    
    return ligand_mol


def mol_to_sdf_block(mol):
    return Chem.MolToMolBlock(mol)


def count_rows(table_name: str, include_existing: bool = False, only_valid: bool = False):
    """Count rows to process (for progress reporting)."""
    query = [
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        WHERE output_pdb_path IS NOT NULL
        """
    ]
    if not include_existing:
        query.append("AND (ligand_sdf IS NULL OR ligand_sdf = '')")
    if only_valid:
        query.append("AND is_valid = TRUE")
    
    sql_query = " ".join(query)
    
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(sql_query)
            return c.fetchone()[0]


def update_ligand_sdf(row_id: int, table_name: str, sdf_block: str):
    """Update a single row with ligand SDF block."""
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                sql.SQL("UPDATE {} SET {} = %s WHERE id = %s").format(
                    sql.Identifier(table_name),
                    sql.Identifier(LIGAND_COLUMN),
                ),
                (sdf_block, row_id),
            )
            conn.commit()


def main():
    args = parse_args()
    section("Ligand SDF Extraction")
    info(f"Target table: {args.table}")
    ensure_schema(args.table)

    # Count total rows for progress reporting
    total_count = count_rows(
        args.table,
        include_existing=args.include_existing,
        only_valid=args.only_valid,
    )
    if args.limit is not None:
        total_count = min(total_count, args.limit)
    info(f"Found {total_count} rows to process")

    processed = 0
    updated = 0

    # Process rows one at a time using generator to minimize memory usage
    for row in fetch_rows_iter(
        args.table,
        limit=args.limit,
        include_existing=args.include_existing,
        only_valid=args.only_valid,
    ):
        row_id, pocket_id, pdb_path, ligand_size, is_valid = row
        processed += 1
        info(f"[{processed}/{total_count}] pocket={pocket_id} pdb={pdb_path}")

        ligand_mol = None
        try:
            ligand_mol = isolate_ligand_mol(pdb_path)
            if ligand_mol is None:
                warn(f"No ligand atoms labeled 'LIG' in {pdb_path}")
                continue
            
            num_atoms = ligand_mol.GetNumAtoms()
            sdf_block = mol_to_sdf_block(ligand_mol)
            # Explicitly delete molecule after converting to SDF to free memory
            del ligand_mol
            
            if args.dry_run:
                info(f"Dry-run: extracted ligand with {num_atoms} atoms")
            else:
                update_ligand_sdf(row_id, args.table, sdf_block)
                updated += 1
                success(f"Stored ligand SDF ({num_atoms} atoms)")
            
            # Clear SDF block from memory after use
            del sdf_block
            
        except Exception as exc:
            error(f"Failed to process {pdb_path}: {exc}")
            # Ensure molecule is deleted even on error
            if ligand_mol is not None:
                del ligand_mol

    success(f"Completed {processed} rows; updated {updated} entries")


if __name__ == "__main__":
    main()

