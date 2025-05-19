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


## Usage

### Generation

```shell
mkdir -p models
python -W ignore yuel_design.py --pocket 2req_pocket.pdb --model test.ckpt --size 15
```

### Training

```shell
mkdir -p models
mkdir -p logs
```

Run trainig:

```shell
python -W ignore train_yuel_design.py --config configs/yuel_design.yml
```

```shell
mkdir -p trajectories
```

# Contact

If you have any questions, please contact me at jianopt@gmail.com
