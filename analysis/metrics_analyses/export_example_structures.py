import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.db_utils import db_connection


def fetch_structures(protein_names):
    protein_names_lower = [name.lower() for name in protein_names]
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, ligand_name, protein_name, pdb
                FROM moad_pockets
                WHERE LOWER(protein_name) = ANY(%s)
                ORDER BY protein_name, id
                """,
                (protein_names_lower,),
            )
            pocket_rows = cur.fetchall()
            if not pocket_rows:
                raise ValueError(f"No pocket found for proteins {protein_names}")

            structures = []
            for pocket_id, pocket_name, ligand_name, protein_name, pocket_pdb in pocket_rows:
                cur.execute(
                    """
                    SELECT name, mol
                    FROM moad_ligands
                    WHERE LOWER(name) = %s
                    """,
                    (ligand_name.lower(),),
                )
                ligand_row = cur.fetchone()
                if ligand_row is None:
                    raise ValueError(f"No ligand entry found for ligand '{ligand_name}'")
                ligand_name_db, ligand_sdf = ligand_row

                cur.execute(
                    """
                    SELECT pdb
                    FROM moad_proteins
                    WHERE LOWER(name) = %s
                    """,
                    (protein_name.lower(),),
                )
                protein_row = cur.fetchone()
                if protein_row is None:
                    raise ValueError(f"No protein entry found for protein '{protein_name}'")
                protein_pdb = protein_row[0]

                structures.append({
                    "pocket_id": pocket_id,
                    "pocket_name": pocket_name,
                    "ligand_name": ligand_name_db,
                    "protein_name": protein_name,
                    "pocket_pdb": pocket_pdb,
                    "ligand_sdf": ligand_sdf,
                    "protein_pdb": protein_pdb,
                })

    return structures


def save_structures(structures_list, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)

    for structures in structures_list:
        protein_name = structures["protein_name"]
        ligand_name = structures['ligand_name']
        pocket_path = output_dir / f"{protein_name}_pocket_{ligand_name}.pdb"
        ligand_path = output_dir / f"{protein_name}_ligand_{ligand_name}.sdf"
        protein_path = output_dir / f"{protein_name}_protein.pdb"

        pocket_path.write_text(structures["pocket_pdb"])
        ligand_path.write_text(structures["ligand_sdf"])
        protein_path.write_text(structures["protein_pdb"])

        print(f"Saved pocket PDB to {pocket_path}")
        print(f"Saved ligand SDF to {ligand_path}")
        print(f"Saved protein PDB to {protein_path}")


def main():
    parser = argparse.ArgumentParser(description="Export example structures for a MOAD protein.")
    parser.add_argument(
        "--protein",
        default="3JQA",
        help="Protein name or comma-separated list of names (default: 3JQA)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "example_structures"),
        help="Directory to save exported structures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    protein_names = [name.strip() for name in args.protein.split(",") if name.strip()]
    if not protein_names:
        raise ValueError("Please provide at least one protein name.")
    structures_list = fetch_structures(protein_names)
    save_structures(structures_list, output_dir)


if __name__ == "__main__":
    main()

