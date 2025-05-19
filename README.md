# YuelDesign: Equivariant 3D-Conditional Diffusion Model for Molecular Linker Design

The design of molecules for flexible protein pockets represents a significant challenge in structure-based drug discovery, as proteins often undergo conformational changes upon ligand binding. While deep learning-based approaches have shown promise in molecular generation, they typically treat protein pockets as rigid structures, limiting their ability to capture the dynamic nature of protein-ligand interactions. Here, we introduce YuelDesign, a novel diffusion-based framework specifically developed to address this challenge. YuelDesign employs a new protein encoding scheme with a fully connected graph representation to encode protein pocket flexibility, a systematic denoising process that refines both atomic properties and coordinates, and a specialized bond reconstruction module tailored for de novo generated molecules. Our results demonstrate that YuelDesign generates molecules with high validity, low large-ring formation rates, favorable drug-likeness, and low synthetic complexity. The generated molecules exhibit diverse chemical functional groups, including some not present in the training set. Redocking analysis reveals that the generated molecules exhibit docking energies comparable to native ligands. Additionally, a detailed analysis of the denoising process shows how the model systematically refines molecular structures through atom type transitions, bond dynamics, and conformational adjustments. Overall, YuelDesign presents a versatile framework for generating novel molecules tailored to flexible protein pockets, with promising implications for drug discovery applications.

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

### Required Parameters

`--model`: Path to the DiffLinker model

`--size`: Linker size, can be:
* A single integer
* Comma-separated integer range
* Path to a size prediction model

### Input Source (choose one)

`--pocket`: Path to the file containing pocket atoms

`--dataset`: Path to the dataset

### Optional Parameters

`--output`: Directory to save generated molecules (default: './')

`--n_samples`: Number of linkers to generate (default: 5)

`--random_seed`: Random seed (default: None)

`--trajectory`: Trajectory directory (default: None)

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
