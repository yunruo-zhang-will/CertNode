#!/usr/bin/env bash

task='Node' # 'Node' or 'Graph'

for model in 'GCN' 'GAT' 'GSAGE' 'GIN' 'APPNP'
do
    for dataset in 'CiteSeer' 'PubMed' 'computers' 'Cora-ML'
    do
        for p_n in 0.95 0.99 0.995 0.9975 #0.995 0.9967 0.9975 0.998 #0.999
        do
            for seed in 42 1994 344 756 1789 911 114 514 1919 810
            do
                python main.py -seed $seed -epochs 200 -lr 0.002 -criterion ce -task $task -dataset $dataset -model $model -p_n $p_n -n_smoothing 10000 -gpuID 0 -output_dir './results_RQ5/'
            done
        done
    done
done

