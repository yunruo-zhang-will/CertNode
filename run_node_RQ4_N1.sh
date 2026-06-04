#!/usr/bin/env bash

task='Node' # 'Node' or 'Graph'

for attack_use_eot in true
do
    for attack_method in 'tdgia' 'gnia' 'grb_injection'
    do
        for dataset in 'CiteSeer'
        do
            for model in 'GCN'
            do
                for Nsmth in 100 200 500 1000 2000
                do
                    python main_attack.py \
                        -epochs 200 \
                        -seed 2020 \
                        -attack_seed 2020 \
                        -lr 0.002 \
                        -criterion ce \
                        -dataset "$dataset" \
                        -model "$model" \
                        -p_n 0.8 \
                        -n_smoothing "$Nsmth" \
                        -gpuID 0 \
                        -output_dir './results_RQ4/N1' \
                        -attack_method "$attack_method" \
                        -attack_use_eot "$attack_use_eot" \
                        -attack_restarts 3 \
                        -attack_n_inject_max 100 \
                        -attack_n_edge_max 5 \
                        -attack_lr 0.01 \
                        -attack_epochs 100 \
                        -attack_rank_samples 32 \
                        -attack_opt_samples 16 \
                        -attack_clean_smoothing "$Nsmth" \
                        -attack_eval_smoothing "$Nsmth" \
                        -attack_sequential_step 0.2
                done
            done
        done
    done
done

