#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --mem=10G
#SBATCH --job-name=notebooks
#SBATCH -o ./out/%j.out

echo ""
echo "Started on node $HOSTNAME at $(date)."
echo ""
echo "--------------------------------------"
echo ""

module load mamba
module load triton/2024.1-gcc cuda/12.2.1

source activate qutip

for file in $1; do
	echo "Executing notebook $file"
	echo ""
	jupyter nbconvert --to notebook --execute "$file" --inplace
	echo "File $file completed at $(date)"
	echo ""
done
echo "Done"
echo ""
echo "--------------------------------------"
echo ""
