#!/usr/bin/env python
"""
Standalone PMDM generator for a single protein pocket PDB and target ligand size.
Loads a PMDM checkpoint, runs sampling, and writes the resulting ligand as an SDF file.
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

pmdm_dir = Path(__file__).resolve().parent
if str(pmdm_dir) not in sys.path:
    sys.path.insert(0, str(pmdm_dir))

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from torch_geometric.data import Batch
from easydict import EasyDict

# Compatibility patches from run_test
try:
    import torch_scatter
except ImportError:
    import torch_scatter_compat
    sys.modules["torch_scatter"] = torch_scatter_compat

try:
    import torch_sparse
except ImportError:
    import torch_sparse_compat
    sys.modules["torch_sparse"] = torch_sparse_compat

# Use our manual implementation of radius_graph
from torch_cluster_compat import radius_graph

from configs.dataset_config import get_dataset_info
from models.epsnet import get_model
from utils.misc import *
from utils.protein_ligand import PDBProtein, ATOM_FAMILIES, ATOM_FAMILIES_ID
from utils.transforms import *
from utils.sample import construct_dataset_pocket
from utils.reconstruct import reconstruct_from_generated
from utils.reconstruct_mdm import mol2smiles
from utils.data import torchify_dict, ProteinLigandData


FOLLOW_BATCH = ["ligand_atom_feature", "protein_atom_feature_full"]
atomic_numbers_crossdock = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])
atomic_numbers_pocket = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34])


def pdb_to_pocket_data(pdb_path):
    import warnings

    warnings.simplefilter("ignore", BiopythonWarning)
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0]
    ptable = Chem.GetPeriodicTable()

    protein_dict = EasyDict(
        {"element": [], "pos": [], "is_backbone": [], "atom_to_aa_type": []}
    )
    for atom in unfold_entities(model, "A"):
        res = atom.get_parent()
        resname = res.get_resname()
        if resname == "MSE":
            resname = "MET"
        if resname not in PDBProtein.AA_NAME_NUMBER:
            continue

        element_symb = atom.element.capitalize()
        if element_symb == "H":
            continue

        pos = torch.tensor(atom.get_coord(), dtype=torch.float)
        protein_dict["element"].append(ptable.GetAtomicNumber(element_symb))
        protein_dict["pos"].append(pos)
        protein_dict["is_backbone"].append(atom.get_name() in ["N", "CA", "C", "O"])
        protein_dict["atom_to_aa_type"].append(PDBProtein.AA_NAME_NUMBER[resname])

    if not protein_dict["pos"]:
        raise ValueError("No protein atoms found in PDB file.")

    protein_dict["element"] = torch.LongTensor(protein_dict["element"])
    protein_dict["pos"] = torch.stack(protein_dict["pos"])
    protein_dict["is_backbone"] = torch.BoolTensor(protein_dict["is_backbone"])
    protein_dict["atom_to_aa_type"] = torch.LongTensor(protein_dict["atom_to_aa_type"])

    ligand_dict = {
        "element": torch.empty(0, dtype=torch.long),
        "pos": torch.empty(0, 3, dtype=torch.float),
        "atom_feature": torch.empty(0, 8, dtype=torch.float),
        "bond_index": torch.empty(2, 0, dtype=torch.long),
        "bond_type": torch.empty(0, dtype=torch.long),
    }

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict, ligand_dict=ligand_dict
    )
    return data


def mol_to_sdf_string(mol):
    from io import StringIO

    sio = StringIO()
    writer = Chem.SDWriter(sio)
    writer.write(mol, confId=0)
    writer.close()
    return sio.getvalue()


def write_xyz_from_generated(pos, element, indicators, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords = pos.detach().cpu().numpy()
    elements = element.detach().cpu().numpy()
    ptable = Chem.GetPeriodicTable()

    atom_symbols = [ptable.GetElementSymbol(int(z)) for z in elements]

    with output_path.open("w") as f:
        f.write(f"{len(coords)}\n")
        f.write("\n")
        for symb, coord in zip(atom_symbols, coords):
            x, y, z_val = float(coord[0]), float(coord[1]), float(coord[2])
            f.write(f"{symb} {x:.3f} {y:.3f} {z_val:.3f}\n")


def load_pmdm(checkpoint_path, device):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if device == "auto":
        # Always try CUDA for performance - don't auto-fallback to CPU
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works with basic operations
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.synchronize()
                device = torch.device("cuda")
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                compute_capability = torch.cuda.get_device_capability(0)
                print(f"[INFO] Using CUDA device: {gpu_name}")
                print(f"[INFO] CUDA version: {cuda_version}")
                print(f"[INFO] PyTorch version: {torch.__version__}")
                print(f"[INFO] GPU compute capability: {compute_capability[0]}.{compute_capability[1]}")
            except Exception as e:
                print(f"[ERROR] CUDA initialization failed: {e}")
                print(f"[ERROR] Cannot use CUDA. Please fix CUDA setup or use --device cpu")
                raise RuntimeError(f"CUDA initialization failed: {e}")
        else:
            print(f"[ERROR] CUDA not available")
            print(f"[ERROR] Please install CUDA-enabled PyTorch")
            print(f"[ERROR] For CPU mode, explicitly use: --device cpu")
            raise RuntimeError("CUDA not available")
    else:
        device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print(f"[ERROR] CUDA device requested but not available")
                print(f"[ERROR] Please install CUDA-enabled PyTorch or use --device cpu")
                raise RuntimeError("CUDA not available")
            print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")

    model = get_model(config.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, device


def generate_raw_pmdm_sample(
    model,
    config,
    device,
    pocket_data,
    ligand_size,
    seed=None,
):
    if seed is not None:
        seed_all(seed)
    else:
        seed_all(config.train.seed)

    dataset_info = get_dataset_info("crossdock_pocket", False)

    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=True)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=True)
    transform = Compose(
        [
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            CountNodesPerGraph(),
            GetAdj(only_prot=True),
        ]
    )
    data = transform(pocket_data)

    protein_atom_feature = data.protein_atom_feature.float()
    protein_atom_feature_full = data.protein_atom_feature_full.float()

    n_steps = config.model.num_diffusion_timesteps
    batch_size = 1
    num_samples = 1

    data_list, _ = construct_dataset_pocket(
        num_samples,
        batch_size,
        dataset_info,
        ligand_size,
        ligand_size,
        None,
        None,
        protein_atom_feature,
        protein_atom_feature_full,
        data.protein_pos,
        data.protein_bond_index,
    )

    datas = data_list[0]
    batch = Batch.from_data_list(datas, follow_batch=FOLLOW_BATCH).to(device)
    with torch.no_grad():
        pos_gen, _, atom_type, _ = model.langevin_dynamics_sample(
            ligand_atom_type=batch.ligand_atom_feature.float(),
            ligand_pos_init=batch.ligand_pos,
            ligand_bond_index=batch.ligand_bond_index,
            ligand_bond_type=batch.ligand_bond_type,
            ligand_num_node=batch.ligand_num_node,
            ligand_batch=batch.ligand_atom_feature_batch,
            protein_atom_type=batch.protein_atom_feature,
            protein_atom_feature_full=batch.protein_atom_feature_full,
            protein_pos=batch.protein_pos,
            protein_bond_index=batch.protein_bond_index,
            protein_backbone_mask=None,
            protein_batch=batch.protein_atom_feature_full_batch,
            num_graphs=batch.num_graphs,
            extend_order=False,
            n_steps=n_steps,
            step_lr=1e-6,
            w_global_pos=1.0,
            w_global_node=1.0,
            w_local_pos=1.0,
            w_local_node=1.0,
            global_start_sigma=float("inf"),
            sampling_type="generalized",
            eta=1.0,
            context=None,
        )

    pos_list = unbatch(pos_gen, batch.ligand_atom_feature_batch)
    atom_list = unbatch(atom_type, batch.ligand_atom_feature_batch)

    pos = pos_list[0]
    atom_feat = atom_list[0]

    new_element = torch.tensor(
        [
            atomic_numbers_crossdock[idx]
            for idx in torch.argmax(atom_feat[:, :8], dim=1)
        ]
    )
    indicators_idx = torch.argmax(atom_feat[:, 8:], dim=1)
    indicators = torch.zeros(
        (pos.size(0), len(ATOM_FAMILIES)), dtype=torch.long
    )
    for row_idx, col_idx in enumerate(indicators_idx):
        indicators[row_idx, col_idx] = 1

    return pos, new_element, indicators


def generate_with_pmdm(
    model,
    config,
    device,
    pocket_data,
    ligand_size,
    max_attempts=20,
    seed=None,
):
    if seed is not None:
        seed_all(seed)
    else:
        seed_all(config.train.seed)

    dataset_info = get_dataset_info("crossdock_pocket", False)

    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=True)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=True)
    transform = Compose(
        [
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            CountNodesPerGraph(),
            GetAdj(only_prot=True),
        ]
    )
    data = transform(pocket_data)
    
    protein_atom_feature = data.protein_atom_feature.float()
    protein_atom_feature_full = data.protein_atom_feature_full.float()

    n_steps = config.model.num_diffusion_timesteps
    batch_size = 5
    num_samples = max_attempts * 2

    data_list, _ = construct_dataset_pocket(
        num_samples,
        batch_size,
        dataset_info,
        ligand_size,
        ligand_size,
        None,
        None,
        protein_atom_feature,
        protein_atom_feature_full,
        data.protein_pos,
        data.protein_bond_index,
    )

    attempts = 0
    start = time.time()

    for batch_idx, datas in enumerate(data_list):
        if attempts >= max_attempts:
            break
        batch = Batch.from_data_list(datas, follow_batch=FOLLOW_BATCH).to(device)
        with torch.no_grad():
            try:
                pos_gen, _, atom_type, _ = model.langevin_dynamics_sample(
                    ligand_atom_type=batch.ligand_atom_feature.float(),
                    ligand_pos_init=batch.ligand_pos,
                    ligand_bond_index=batch.ligand_bond_index,
                    ligand_bond_type=batch.ligand_bond_type,
                    ligand_num_node=batch.ligand_num_node,
                    ligand_batch=batch.ligand_atom_feature_batch,
                    protein_atom_type=batch.protein_atom_feature,
                    protein_atom_feature_full=batch.protein_atom_feature_full,
                    protein_pos=batch.protein_pos,
                    protein_bond_index=batch.protein_bond_index,
                    protein_backbone_mask=None,
                    protein_batch=batch.protein_atom_feature_full_batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False,
                    n_steps=n_steps,
                    step_lr=1e-6,
                    w_global_pos=1.0,
                    w_global_node=1.0,
                    w_local_pos=1.0,
                    w_local_node=1.0,
                    global_start_sigma=float("inf"),
                    sampling_type="generalized",
                    eta=1.0,
                    context=None,
                )

                pos_list = unbatch(pos_gen, batch.ligand_atom_feature_batch)
                atom_list = unbatch(atom_type, batch.ligand_atom_feature_batch)

                for mol_idx, (pos, atom_feat) in enumerate(zip(pos_list, atom_list)):
                    if attempts >= max_attempts:
                        break
                    attempts += 1
                    try:
                        pos = pos.detach().cpu()
                        atom_feat = atom_feat.detach().cpu()
                        new_element = torch.tensor(
                            [
                                atomic_numbers_crossdock[idx]
                                for idx in torch.argmax(atom_feat[:, :8], dim=1)
                            ]
                        )
                        indicators_idx = torch.argmax(atom_feat[:, 8:], dim=1)
                        indicators = torch.zeros(
                            (pos.size(0), len(ATOM_FAMILIES)), dtype=torch.long
                        )
                        for row_idx, col_idx in enumerate(indicators_idx):
                            indicators[row_idx, col_idx] = 1

                        gmol = reconstruct_from_generated(pos, new_element, indicators)
                        g_smile = mol2smiles(gmol)
                        
                        if g_smile is None:
                            continue
                        
                        if "." in g_smile:
                            continue
                        
                        duration = time.time() - start
                        return gmol, attempts, duration
                    except Exception as e:
                        continue
            except Exception as e:
                attempts += 1
                continue

    duration = time.time() - start
    return None, attempts, duration


def main():
    parser = argparse.ArgumentParser(description="PMDM generator")
    parser.add_argument("--protein-pdb", required=True, help="Path to pocket PDB file")
    parser.add_argument("--ligand-size", required=True, type=int, help="Target ligand size")
    parser.add_argument("--checkpoint", default="500.pt", help="PMDM checkpoint path")
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

    # Load pocket data and model once
    pocket_data = pdb_to_pocket_data(args.protein_pdb)
    model, config, device = load_pmdm(args.checkpoint, args.device)
    
    # Set initial seed only once at the start
    if args.seed is not None:
        final_seed = args.seed
    else:
        final_seed = int(time.time() * 1000000) % (2**31)
        print(f"[INFO] No seed provided, using time-based seed: {final_seed}")
    
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)
    
    if args.output_xyz is not None:
        output_path = Path(args.output_xyz)
        print("[INFO] Generating single ligand XYZ without retries...")
        pos, new_element, indicators = generate_raw_pmdm_sample(
            model=model,
            config=config,
            device=device,
            pocket_data=pocket_data,
            ligand_size=args.ligand_size,
            seed=final_seed,
        )
        write_xyz_from_generated(pos, new_element, indicators, output_path)
        print(f"[SUCCESS] Generated ligand XYZ and saved to: {output_path}")
    else:
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
            
            print(f"[INFO] Generating molecule {i}/{args.num_molecules}...")
            
            # Keep retrying until valid (up to external retry limit)
            external_attempt = 0
            max_external_retries = 10
            
            while external_attempt < max_external_retries:
                # Use different random seed for each attempt
                current_seed = np.random.randint(0, 2**31)
                if external_attempt > 0:
                    print(f"[INFO] Retry attempt {external_attempt + 1} with seed={current_seed}")
                else:
                    print(f"[INFO] Attempt {external_attempt + 1} with seed={current_seed}")
                
                mol, attempts, duration = generate_with_pmdm(
                    model=model,
                    config=config,
                    device=device,
                    pocket_data=pocket_data,
                    ligand_size=args.ligand_size,
                    max_attempts=args.max_attempts,
                    seed=current_seed,
                )

                if mol is not None:
                    sdf = mol_to_sdf_string(mol)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(sdf)
                    print(f"[SUCCESS] Molecule {i} generated in {attempts} attempts ({duration:.2f}s)")
                    print(f"[INFO] Saved SDF to: {output_path}")
                    success_count += 1
                    break
                else:
                    external_attempt += 1
                    if external_attempt < max_external_retries:
                        print(f"[WARN] Molecule {i} failed after {attempts} attempts (external retry {external_attempt}/{max_external_retries}), retrying with new seed...")
            
            if mol is None:
                print(f"[ERROR] Molecule {i} failed after {max_external_retries} external retries ({attempts} internal attempts in last retry)", file=sys.stderr)
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

