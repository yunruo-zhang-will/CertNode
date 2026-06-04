#!/usr/bin/env bash

task='Node' # 'Node' or 'Graph'

for dataset in 'CiteSeer' 'PubMed' 'computers' 'Cora-ML'
do
    for model in 'GCN' 'GAT' 'GSAGE' 'GIN' 'APPNP'
    do
        for p_n in 0.995 0.9967 0.9975 
        do
            python main.py -epochs 200 -lr 0.002 -criterion ce -task $task -dataset $dataset -model $model -p_n $p_n -n_smoothing 10000 -gpuID 0 -output_dir './results_RQ1/'
        done
    done
done

