# YuelDesign: Equivariant 3D-Conditional Diffusion Model for Molecular Linker Design


## Environment Setup

installing all the necessary packages:

```shell
rdkit
pdb-tools
biopython
imageio
networkx
pytorch
pytorch-lightning
scipy
scikit-learn
tqdm
wandb
```


## Generation

1. Download the models

```shell
mkdir -p models
wget https://zenodo.org/records/15467850/files/moad.ckpt?download=1 -O models/moad.ckpt
```

2. Generation

```shell
mkdir -p models
python -W ignore yuel_design.py --pocket 2req_pocket.pdb --model test.ckpt --size 15
python yuel_design.py \
    --pocket 2req_pocket.pdb \
    --model models/moad.ckpt \
    --output 2req_generation \
    --random_seed 123 \
    --size 15 \
    --n_samples 50
```

Parameters:

Required:
- `--model`: Path to the DiffLinker model
- `--size`: Linker size, can be:
  - A single integer
  - Comma-separated integer range
  - Path to a size prediction model

Input Source (choose one):
- `--pocket`: Path to the file containing pocket atoms
- `--dataset`: Path to the dataset

Optional:
- `--output`: Directory to save generated molecules (default: './')
- `--n_samples`: Number of linkers to generate (default: 5)
- `--random_seed`: Random seed (default: None)
- `--trajectory`: Trajectory directory (default: None)

## Training

1. Download the trainingdata

```shell
mkdir -p datasets
wget https://zenodo.org/records/15467850/files/MOAD_train.pkl?download=1 -O datasets/MOAD_train.pkl
wget https://zenodo.org/records/15467850/files/MOAD_val.pkl?download=1 -O datasets/MOAD_val.pkl
wget https://zenodo.org/records/15467850/files/MOAD_test.pkl?download=1 -O datasets/MOAD_test.pkl
```

2. Training

```shell
mkdir -p models
mkdir -p logs
python -W ignore train_yuel_design.py --config configs/train_moad.yml
```

```shell
mkdir -p trajectories
```

# Contact

If you have any questions, please contact me at jianopt@gmail.com
