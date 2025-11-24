# YuelDesign: A Diffusion-Based Framework for Designing Molecules in Flexible Protein Pockets

Designing molecules for flexible protein pockets poses a significant challenge in structure-based drug discovery, as proteins often undergo conformational changes upon ligand binding. While deep learning-based methods have shown promise in molecular generation, they typically treat protein pockets as rigid structures, limiting their ability to capture the dynamic nature of protein-ligand interactions. Here, we present YuelDesign, a diffusion-based framework that jointly models protein-ligand complexes with continuous atomic coordinates and discrete atom types. YuelDesign employs E3former, an Evoformer-based equivariant model, to encode protein pockets and ligands while maintaining rotational and translational equivariance. The framework incorporates dual diffusion processes, an Elucidated Diffusion Model (EDM) for coordinates and a Discrete Denoising Diffusion Probabilistic Model (D3PM) for ligand atom types, enabling iterative refinement of both geometry and chemical identity. Our results demonstrate that YuelDesign generates molecules with favorable drug-likeness, low synthetic complexity, diverse chemical functional groups, and docking energies comparable to native ligands. YuelDesign presents a versatile framework for designing drugs in flexible protein pockets, with promising implications for drug discovery applications.

## Environment Setup

Install all the necessary packages:

```shell
pip install rdkit
pip install pdb-tools
pip install biopython
pip install imageio
pip install networkx
pip install scipy
pip install scikit-learn
pip install tqdm
pip install wandb
pip install pyyaml
pip install pytorch-lightning
pip install psycopg2-binary
```

Note: 
- Install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/) based on your system and CUDA version.
- Install PostgreSQL server separately. On Ubuntu/Debian: `sudo apt-get install postgresql`, on macOS: `brew install postgresql`

### Wandb Configuration (Optional, for Training)

If you plan to use wandb for logging training metrics, create a `wandb_api_key.txt` file in the project root directory with your wandb API key:

```shell
echo "your_wandb_api_key_here" > wandb_api_key.txt
```

You can get your API key from [wandb.ai/settings](https://wandb.ai/settings).


## Generation

### 1. Download the Model Checkpoint

```shell
mkdir -p checkpoints
wget https://zenodo.org/records/15467850/files/moad.ckpt?download=1 -O checkpoints/moad.ckpt
```

### 2. Generate Molecules

Generate a single molecule:

```shell
python yuel_design.py \
    --input 2req_pocket.pdb \
    --checkpoint checkpoints/moad.ckpt \
    --output 2req_generated.pdb \
    --ligand_size 15 \
    --seed 123
```

Generate multiple molecules:

```shell
python yuel_design.py \
    --input 2req_pocket.pdb \
    --checkpoint checkpoints/moad.ckpt \
    --output-prefix 2req_generation \
    --ligand_size 15 \
    --num_molecules 50 \
    --seed 123
```

### Command-Line Arguments

#### Required Parameters

- `-i, --input`: Path to the input protein pocket PDB file

#### Optional Parameters

- `--checkpoint`: Path to the model checkpoint file (if not specified, automatically detects the latest checkpoint in `checkpoints/` directory)
- `-o, --output`: Output PDB file path (required when generating a single molecule, i.e., `-n=1`)
- `--output-prefix`: Output file prefix for multiple molecules (required when `-n > 1`). Generated files will be named as `{prefix}_1.pdb`, `{prefix}_2.pdb`, etc.
- `--ligand_size`: Number of ligand atoms to generate (if not specified, will be auto-estimated from receptor size)
- `-n, --num_molecules`: Number of molecules to generate (default: 1)
- `--seed`: Random seed for reproducibility (default: None, uses random seed)
- `--device`: Device to use - `auto`, `cpu`, or `cuda` (default: `auto`)
- `--save_trajectory`: Path to save diffusion trajectory as multi-model PDB file (optional, supports `{index}` placeholder for multiple molecules)
- `--log`: Path to save validation statistics log file (optional, supports `{index}` placeholder for multiple molecules)
- `--max_attempts`: Maximum number of attempts per molecule to generate a valid ligand (default: 20)

## Training

### Prerequisites

Training requires:
1. **PostgreSQL database**: Install and set up a PostgreSQL database server
2. **Database configuration**: Create a database configuration file at `src/db_config` with the following format:
   ```
   DB_NAME=your_database_name
   DB_USER=your_username
   DB_HOST=localhost
   DB_PORT=5432
   ```
3. **MOAD dataset**: Download and prepare the MOAD dataset (see below)
4. **Initialize database tables**: Load MOAD data into the database (see below)

### MOAD Dataset Setup

1. **Download MOAD dataset** (if not already downloaded):
   ```shell
   cd data/MOAD
   wget https://zenodo.org/record/13375913/files/every_part_a.zip
   wget https://zenodo.org/record/13375913/files/every_part_b.zip
   unzip every_part_a.zip
   unzip every_part_b.zip
   ```

2. **Prepare processed data directories**:
   The MOAD data should be organized in the following structure:
   ```
   data/MOAD/
   ├── processed/
   │   ├── proteins/      # PDB files named as {pdb_code}_protein.pdb
   │   ├── ligands/       # MOL files named as {pdb_code}_{residue_name}{residue_id}_{chain_id}_{i}.mol
   │   └── pockets/       # PDB files named as {pdb_code}_{residue_name}{residue_id}_{chain_id}_{i}_pocket.pdb
   ```
   
   See `data/MOAD/README.md` for detailed instructions on data preparation.

3. **Initialize database tables**:
   
   Before running the initialization script, edit `data/MOAD/init_moad.py` and update the directory paths at the bottom of the file (around lines 274-276) to point to your processed data directories:
   ```python
   proteins_dir = "/absolute/path/to/data/MOAD/processed/proteins"
   ligands_dir = "/absolute/path/to/data/MOAD/processed/ligands"
   pockets_dir = "/absolute/path/to/data/MOAD/processed/pockets"
   ```
   
   Then run the initialization script from the project root:
   ```shell
   python data/MOAD/init_moad.py
   ```
   
   This script will:
   - Create database tables: `moad_proteins`, `moad_ligands`, and `moad_pockets`
   - Load all protein, ligand, and pocket files from the processed directories
   - Create necessary indexes for efficient queries
   - Display statistics about the loaded data
   
4. **Split dataset into train/val/test sets**:
   
   After initializing the database, run the split script from the project root to add the `split` column and assign train/val/test labels:
   ```shell
   python data/MOAD/split_moad.py
   ```
   
   This script will:
   - Add a `split` column to the `moad_pockets` table if it doesn't exist
   - Randomly assign splits: ~80% train, ~15% test, 50 samples for validation
   - Display a summary of the split distribution
   
   See `DATABASE_SCHEMA.md` for details about the database schema.

### Training Steps

1. **Configure training parameters** in `src/e2efinal/config.py`:
   - `exp_name`: Experiment name
   - `batch_size`: Batch size for training
   - `n_epochs`: Number of training epochs
   - `lr`: Learning rate
   - `checkpoints`: Directory to save model checkpoints
   - `logs`: Directory to save training logs
   - Model architecture parameters (E3former blocks, hidden dimensions, etc.)

2. Run training:

```shell
mkdir -p checkpoints
mkdir -p logs
python src/e2efinal/train.py
```

Training will automatically:
- Load data from the database using `E2EDataset`
- Use `E2EModel` (E3former-based diffusion model)
- Save checkpoints to the `checkpoints/` directory
- Log training metrics (can be configured for wandb)

3. Resume training from a checkpoint:

Edit `src/e2efinal/config.py` and set:
```python
resume = "experiment_name"  # Name of the experiment directory in checkpoints/
```

Then run the training script again.

## Model Architecture

YuelDesign uses a dual diffusion process for end-to-end ligand generation:

1. **Continuous Diffusion (EDM)**: Models atomic coordinates using an Elucidated Diffusion Model with polynomial noise schedule
2. **Discrete Diffusion (D3PM)**: Models ligand atom types using a Discrete Denoising Diffusion Probabilistic Model with uniform transition matrices

The model backbone is **E3former**, an Evoformer-based equivariant neural network that:
- Maintains rotational and translational equivariance
- Encodes protein-ligand complexes with continuous coordinates and discrete atom types
- Uses attention mechanisms for sequence and pairwise interactions

Key features:
- Joint modeling of coordinates and atom types
- Equivariant architecture ensures physical consistency
- Anchor-based diffusion (protein CA atoms remain fixed during generation)

## Output Format

Generated structures are saved as PDB files containing:
- **Receptor atoms**: All protein atoms from the input pocket (excluding hydrogen atoms)
- **Ligand atoms**: Generated ligand atoms saved as HETATM records with residue name "LIG" and chain ID "B"

If trajectory saving is enabled (`--save_trajectory`), a multi-model PDB file is generated where each MODEL record represents a diffusion timestep.

Validation statistics (if `--log` is specified) include:
- Total number of generation attempts
- Success/failure status
- Failure reasons (ring size violations, fragmentation, kekulization errors)

## Validation

The generated ligands are validated for:
- **Connectivity**: Ensures the ligand is a single connected molecule
- **Ring sizes**: Rejects molecules with 3-membered, 4-membered, or >6-membered rings
- **Kekulization**: Ensures the molecule can be properly kekulized by RDKit

If validation fails, the model automatically retries with a new random seed (up to `--max_attempts` times).

## Contact

If you have any questions, please contact me at jianopt@gmail.com
