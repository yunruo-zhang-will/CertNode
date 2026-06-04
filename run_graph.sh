#!/usr/bin/env bash

task='Graph' # 'Node' or 'Graph'

for dataset in 'Mutagenicity' 'PROTEINS' 'DD' 'AIDS'
do
    for model in 'GCN' 'GAT' 'GSAGE' 'GIN' 'APPNP'
    do
        for p_n in 0.96 0.975 0.98
        do
            python main.py -epochs 200 -lr 0.002 -criterion ce -task $task -dataset $dataset -model $model -p_n $p_n -n_smoothing 10000 -output_dir './results_graph/'
        done
    done
done

