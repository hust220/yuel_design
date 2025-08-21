wd=$(cd $(dirname $0); pwd)

irow=1
while true; do
    id=$RANDOM
    script=${wd}/docking/${id}.sh
    cat <<! >${script}
#!/bin/bash 
#SBATCH --mem=8GB 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00 
#SBATCH --account=nxd338_nih
#SBATCH --partition=mgc-nih
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --job-name=docking
cd ${wd}

module load miniconda/3
conda activate torch

python docking_worker.py --batch 20 --n 1

rm ${script}
!
    irow=$((irow+1))
    while true; do
        n=$(squeue -u $USER | grep $USER | grep nih | perl -lane '$n+=$F[6];END{print $n}')
        if [[ ${n} -lt 150 ]]; then
            echo \[${irow}\] Submitting Task ${pdbid} ...
            sbatch ${script}
            break
        else
            echo Waiting ...
            sleep 10
        fi
    done
done

