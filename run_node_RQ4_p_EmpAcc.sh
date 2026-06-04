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
                for p_n in 0.4 0.6 0.8 0.9 0.95
                do
                    python main_attack.py \
                        -epochs 200 \
                        -seed 2020 \
                        -attack_seed 2020 \
                        -lr 0.002 \
                        -criterion ce \
                        -dataset "$dataset" \
                        -model "$model" \
                        -p_n "$p_n" \
                        -n_smoothing 1000 \
                        -gpuID 0 \
                        -output_dir './results_RQ4/p/EmpAcc' \
                        -attack_method "$attack_method" \
                        -attack_use_eot "$attack_use_eot" \
                        -attack_restarts 3 \
                        -attack_n_inject_max 100 \
                        -attack_n_edge_max 5 \
                        -attack_lr 0.01 \
                        -attack_epochs 100 \
                        -attack_rank_samples 32 \
                        -attack_opt_samples 16 \
                        -attack_clean_smoothing 1000 \
                        -attack_eval_smoothing 1000 \
                        -attack_sequential_step 0.2
                done
            done
        done
    done
done

