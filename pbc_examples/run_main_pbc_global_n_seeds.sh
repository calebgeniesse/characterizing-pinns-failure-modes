#!/bin/bash

# module load python
# mamba activate /global/cfs/cdirs/m636/geniesse/envs/pinns


for beta in 1 50 ; do 

    _cmd="python3 main_pbc_global_n_seeds.py --beta=${beta} --lr=1.0 --n_seeds=100 --visualiz=True --save_model=True"; 

    # echo ""
    echo $_cmd
    # echo ""
    # eval $_cmd
    # echo ""

done


