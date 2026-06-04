#!/usr/bin/env bash

task='Node' # 'Node' or 'Graph'

for dataset in 'CiteSeer' 'PubMed' 'Cora-ML' 'computers'
do
    for model in 'GCN' 'GAT' 'GSAGE' 'GIN' 'APPNP'
    do
        for p_n in 0.95 0.99 0.995
        do
            for n_s in 100 200 500 1000 2000 5000 10000 #20000 #50000 100000
            do
                python main.py -epochs 200 -lr 0.002 -criterion ce -task $task -dataset $dataset -model $model -p_n $p_n -n_smoothing $n_s -gpuID 0 -output_dir './results_RQ4/N2/'
            done
        done
    done
done

