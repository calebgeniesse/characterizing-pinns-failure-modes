#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=shared
#SBATCH --account=m636
#SBATCH -o logs/%j.out # STDOUT
#SBATCH -e logs/%j.err # STDERR


###############################################################################
### setup here
###############################################################################

pwd
hostname
date
echo starting job...
source ~/.bashrc

module load python
mamba activate /global/cfs/cdirs/m636/geniesse/envs/pinns


export PYTHONUNBUFFERED=1



###############################################################################
### run scripts here
###############################################################################

for beta in 1 50 ; do 

    _cmd="python3 main_pbc_global_n_seeds.py --beta=${beta} --lr=1.0 --n_seeds=100 --visualize=True --save_model=True"; 

    echo ""
    echo $_cmd
    echo ""
    eval $_cmd
    echo ""

done



###############################################################################
### finish
###############################################################################

wait
date
