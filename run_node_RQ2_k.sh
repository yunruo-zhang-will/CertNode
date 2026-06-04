#!/usr/bin/env bash
# This exp compares attacks with eot and those without

set -euo pipefail

for attack_use_eot in true
do
    for attack_method in 'tdgia' 'gnia' 'grb_injection'
    do
        for dataset in 'CiteSeer'
        do
            for model in 'GCN'
            do
                for p_n in 0.8
                do
                    for Nnode in 20 40 60 80 100
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
                            -output_dir './results_RQ2/eval_k' \
                            -attack_method "$attack_method" \
                            -attack_use_eot "$attack_use_eot" \
                            -attack_restarts 3 \
                            -attack_n_inject_max "$Nnode" \
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
done
