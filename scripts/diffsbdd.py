#!/usr/bin/env python
"""
Simple DiffSBDD CLI utility.
Given a protein pocket PDB file, a reference ligand SDF, and a target ligand size,
run DiffSBDD once (with retries) and save the resulting molecule as an SDF file.
"""
import argparse
import sys
from pathlib import Path
import time

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

diffsbdd_dir = Path(__file__).resolve().parent
if str(diffsbdd_dir) not in sys.path:
    sys.path.insert(0, str(diffsbdd_dir))

import numpy as np
import utils

try:
    import torch
    from rdkit import Chem
    from lightning_modules import LigandPocketDDPM
    from analysis.molecule_builder import process_molecule
except ImportError as e:
    print(f"[ERROR] Failed to import DiffSBDD dependencies: {e}", file=sys.stderr)
    raise


def mol_to_sdf_string(mol):
    from io import StringIO
    sdf_string = StringIO()
    writer = Chem.SDWriter(sdf_string)
    writer.write(mol, confId=0)
    writer.close()
    return sdf_string.getvalue()


def load_model(checkpoint_path, device):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = LigandPocketDDPM.load_from_checkpoint(
        checkpoint_path, 
        map_location=device,
        weights_only=False
    )
    model = model.to(device)
    model.eval()
    return model, device


def generate_molecule(
    model,
    device,
    pocket_pdb,
    ref_ligand,
    ligand_size,
    max_attempts=20,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ligand_tensor = torch.ones(max_attempts, dtype=torch.long, device=device) * ligand_size
    start = time.time()

    for attempt in range(max_attempts):
        try:
            batch = model.generate_ligands(
                pocket_pdb,
                1,
                pocket_ids=None,
                ref_ligand=ref_ligand,
                num_nodes_lig=ligand_tensor[attempt : attempt + 1],
                sanitize=False,
                largest_frag=False,
                relax_iter=0,
                timesteps=None,
                n_nodes_bias=0,
                n_nodes_min=0,
            )
            if not batch:
                continue
            mol = batch[0]
            if mol is None:
                continue

            processed = process_molecule(
                mol,
                sanitize=False,
                relax_iter=0,
                largest_frag=False,
            )
            if processed is None:
                continue

            if processed.GetNumAtoms() <= 0:
                continue

            duration = time.time() - start
            return processed, attempt + 1, duration
        except Exception:
            continue

    duration = time.time() - start
    return None, max_attempts, duration


def main():
    parser = argparse.ArgumentParser(description="DiffSBDD generator")
    parser.add_argument("--protein-pdb", required=True, help="Path to pocket PDB file")
    parser.add_argument("--ref-ligand-sdf", required=True, help="Path to reference ligand SDF file")
    parser.add_argument("--ligand-size", type=int, required=True, help="Target ligand atom count")
    parser.add_argument("--checkpoint", default="checkpoints/moad_fullatom_cond.ckpt", help="DiffSBDD checkpoint path")
    parser.add_argument("--device", default="auto", help="Torch device (default: auto)")
    parser.add_argument("--max-attempts", type=int, default=20, help="Max sampling attempts")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-sdf", default=None, help="Where to save generated SDF (single molecule)")
    parser.add_argument("--output-prefix", default=None, help="Output file prefix for multiple molecules (used with -n)")
    parser.add_argument("--output-xyz", default=None, help="Where to save a single raw ligand as XYZ (no retries, positions + atom types)")
    parser.add_argument("-n", "--num-molecules", type=int, default=1, help="Number of molecules to generate (default: 1)")
    args = parser.parse_args()

    # Validate arguments
    if args.output_xyz is not None:
        if args.num_molecules != 1:
            print("[ERROR] --output-xyz only supports generating a single molecule (use -n 1)", file=sys.stderr)
            sys.exit(1)
    else:
        if args.num_molecules > 1 and args.output_prefix is None:
            print("[ERROR] --output-prefix is required when generating multiple molecules (-n > 1)", file=sys.stderr)
            sys.exit(1)
        if args.num_molecules == 1 and args.output_sdf is None and args.output_prefix is None:
            print("[ERROR] Either --output-sdf or --output-prefix must be specified", file=sys.stderr)
            sys.exit(1)

    # Load model once
    model, device = load_model(args.checkpoint, args.device)
    
    base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**31)

    # Raw XYZ mode: single-shot generation without retries
    if args.output_xyz is not None:
        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        ligand_tensor = torch.ones(1, dtype=torch.long, device=device) * args.ligand_size
        start = time.time()

        raw_batch = model.generate_ligands(
            args.protein_pdb,
            1,
            pocket_ids=None,
            ref_ligand=args.ref_ligand_sdf,
            num_nodes_lig=ligand_tensor,
            sanitize=False,
            largest_frag=False,
            relax_iter=0,
            timesteps=None,
            n_nodes_bias=0,
            n_nodes_min=0,
            return_raw=True,
        )

        if not raw_batch:
            print("[ERROR] Raw generation returned no samples", file=sys.stderr)
            sys.exit(1)

        positions, atom_types = raw_batch[0]
        positions = positions.detach().cpu().numpy()
        atom_types = atom_types.detach().cpu().numpy()

        atom_decoder = model.lig_type_decoder
        atom_symbols = [atom_decoder[int(a)] for a in atom_types]

        output_path = Path(args.output_xyz)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        utils.write_xyz_file(positions, atom_symbols, str(output_path))

        duration = time.time() - start
        print(f"[SUCCESS] Raw ligand XYZ generated in {duration:.2f}s")
        print(f"[INFO] Saved XYZ to: {output_path}")
        return

    success_count = 0
    failed_count = 0

    # Generate multiple molecules
    for i in range(1, args.num_molecules + 1):
        # Determine output path
        if args.num_molecules > 1:
            output_path = Path(f"{args.output_prefix}_{i}.sdf")
        elif args.output_prefix:
            output_path = Path(f"{args.output_prefix}_1.sdf")
        else:
            output_path = Path(args.output_sdf)
        
        # Use different seed for each molecule
        current_seed = base_seed + i if args.seed is not None else None
        
        print(f"[INFO] Generating molecule {i}/{args.num_molecules}...")
        
        # Keep retrying until valid (up to external retry limit)
        external_attempt = 0
        max_external_retries = 10
        
        while external_attempt < max_external_retries:
            if external_attempt > 0:
                current_seed = np.random.randint(0, 2**31)
                print(f"[INFO] Retry attempt {external_attempt + 1} with seed={current_seed}")
            
            mol, attempts, duration = generate_molecule(
                model=model,
                device=device,
                pocket_pdb=args.protein_pdb,
                ref_ligand=args.ref_ligand_sdf,
                ligand_size=args.ligand_size,
                max_attempts=args.max_attempts,
                seed=current_seed,
            )

            if mol is not None:
                sdf_string = mol_to_sdf_string(mol)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(sdf_string)
                print(f"[SUCCESS] Molecule {i} generated in {attempts} attempts ({duration:.2f}s)")
                print(f"[INFO] Saved SDF to: {output_path}")
                success_count += 1
                break
            else:
                external_attempt += 1
                if external_attempt < max_external_retries:
                    print(f"[WARN] Molecule {i} failed after {attempts} attempts, retrying with new seed...")
        
        if mol is None:
            print(f"[ERROR] Molecule {i} failed after {max_external_retries} external retries", file=sys.stderr)
            failed_count += 1
    
    # Summary
    if args.num_molecules > 1:
        print(f"\n[SUMMARY] Generated {success_count}/{args.num_molecules} molecules successfully")
        if failed_count > 0:
            print(f"[WARNING] {failed_count} molecules failed to generate", file=sys.stderr)
        
        if success_count == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()

