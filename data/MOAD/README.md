# How to prepare Pockets dataset 

Download Binding MOAD:
```
wget https://zenodo.org/record/13375913/files/every_part_a.zip
wget https://zenodo.org/record/13375913/files/every_part_b.zip
unzip every_part_a.zip
unzip every_part_b.zip
```

Combine ligands to one file:
```
python -W ignore data/MOAD/combine_ligands.py data/MOAD/processed/ligands data/MOAD/processed/generated_conformers.sdf
```

Prepare dataset:
```
python -W ignore prepare_dataset1.py --sdf processed/conformers.sdf --proteins processed/proteins --out processed/MOAD.pkl
```

Final filtering and train/val/test split:
```
python -W ignore ../split_dataset.py processed/MOAD.pkl
```
