#!/bin/bash

for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 15.0 30.0 50.0 70.0; do
    _cmd="python3 main_pbc.py --beta=$i --visualiz=True --save_model=True --seed=0 --lr=0.0001"
    echo "$" $_cmd
    eval $_cmd
done

for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 15.0 30.0 50.0 70.0; do
    _cmd="python3 main_pbc.py --beta=$i --visualiz=True --save_model=True --seed=123 --lr=0.0001"
    echo "$" $_cmd
    eval $_cmd
done

for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 15.0 30.0 50.0 70.0; do
    _cmd="python3 main_pbc.py --beta=$i --visualiz=True --save_model=True --seed=123456 --lr=0.0001"
    echo "$" $_cmd
    eval $_cmd
done

for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 15.0 30.0 50.0 70.0; do
    _cmd="python3 main_pbc.py --beta=$i --visualiz=True --save_model=True --seed=2023 --lr=0.0001"
    echo "$" $_cmd
    eval $_cmd
done