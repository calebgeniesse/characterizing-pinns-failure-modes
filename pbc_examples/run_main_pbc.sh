#!/bin/bash

for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 15.0 30.0 50.0 70.0; do
    for j in 0 123 123456 2023 54321; do
        for k in 1.0 0.1 0.01 0.001 0.0001; do
            _cmd="python3 main_pbc.py --beta=$i --visualiz=True --save_model=True --seed=$j --lr=$k"
            echo "$" $_cmd
            eval $_cmd
        done
    done
done

echo "Done"
