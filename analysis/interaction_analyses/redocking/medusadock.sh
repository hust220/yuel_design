receptor=$1
ligand_sdf=$2
pocket_mol=$3
outfile=$4

wd=$(cd $(dirname $0); pwd)

module load miniconda/3 medusa
conda activate bio

# pymol command to convert pocket_mol to pocket_mol2
pocket_mol2=$(mktemp /tmp/XXXXXX.mol2)
pymol -cq -d "load $pocket_mol; save $pocket_mol2"

# pymol command to convert ligand_sdf to ligand_mol2
ligand_mol2=$(mktemp /tmp/XXXXXX.mol2)
pymol -cq -d "load $ligand_sdf; save $ligand_mol2"

# cp $receptor $wd/rec.pdb
# cp $ligand_mol2 $wd/lig.mol2
# cp $pocket_mol2 $wd/pocket.mol2

medusa dock -p $MEDUSA_PARAMETER -i $receptor -m $ligand_mol2 -M $pocket_mol2 -o $outfile -R
