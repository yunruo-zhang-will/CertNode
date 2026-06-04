#!/usr/bin/env bash
# This exp eval defences under attacks

set -euo pipefail

EPOCH=200
patience=30
lr=0.002
attackNedge=5
attacklr=0.01
attackepc=100

for attackNnode in 20 40 60 80 100
do
    for attack_method in 'tdgia' 'gnia' 'grb_injection'
    do
        for dataset in 'CiteSeer'
        do
            for model in 'GCN'
            do
                # Smoothed model p=0.8
                python main_attack.py \
                    -epochs "$EPOCH" \
                    -seed 2020 \
                    -attack_seed 2021 \
                    -patience "$patience" \
                    -lr "$lr" \
                    -criterion ce \
                    -dataset "$dataset" \
                    -model "$model" \
                    -p_n 0.8 \
                    -n_smoothing 1000 \
                    -output_dir './results_RQ2/EmpAcc/smooth0.8' \
                    -model_dir './models_smooth' \
                    -attack_method "$attack_method" \
                    -attack_use_eot false \
                    -attack_restarts 3 \
                    -attack_n_inject_max "$attackNnode" \
                    -attack_n_edge_max "$attackNedge" \
                    -attack_lr "$attacklr" \
                    -attack_epochs "$attackepc" \
                    -attack_rank_samples 32 \
                    -attack_opt_samples 16 \
                    -attack_clean_smoothing 1000 \
                    -attack_eval_smoothing 1000 \
                    -attack_sequential_step 0.2

                # Smoothed model p=0.9
                python main_attack.py \
                    -epochs "$EPOCH" \
                    -seed 2020 \
                    -attack_seed 2021 \
                    -patience "$patience" \
                    -lr "$lr" \
                    -criterion ce \
                    -dataset "$dataset" \
                    -model "$model" \
                    -p_n 0.9 \
                    -n_smoothing 1000 \
                    -output_dir './results_RQ2/EmpAcc/smooth0.9' \
                    -model_dir './models_smooth' \
                    -attack_method "$attack_method" \
                    -attack_use_eot false \
                    -attack_restarts 3 \
                    -attack_n_inject_max "$attackNnode" \
                    -attack_n_edge_max "$attackNedge" \
                    -attack_lr "$attacklr" \
                    -attack_epochs "$attackepc" \
                    -attack_rank_samples 32 \
                    -attack_opt_samples 16 \
                    -attack_clean_smoothing 1000 \
                    -attack_eval_smoothing 1000 \
                    -attack_sequential_step 0.2
                
                # GNNCuard
                python main_gnnguard.py \
                    -dataset "$dataset" \
                    -model "$model" \
                    -attack_method "$attack_method" \
                    -epochs "$EPOCH" \
                    -eval_interval 5 \
                    -patience "$patience" \
                    -lr "$lr" \
                    -criterion ce \
                    -force_training \
                    -output_dir './results_RQ2/EmpAcc/gnnguard' \
                    -model_root './models_gnnguard' \
                    -attack_n_inject_max "$attackNnode" \
                    -attack_n_edge_max "$attackNedge" \
                    -attack_lr "$attacklr" \
                    -attack_epochs "$attackepc" \
                    -attack_sequential_step 0.2 \
                    -attack_quiet
                
                # Adv Train
                python main_adv.py \
                    -dataset "$dataset" \
                    -model "$model" \
                    -train_attack_method "$attack_method" \
                    -eval_attack_method "$attack_method" \
                    -epochs "$EPOCH" \
                    -eval_interval 5 \
                    -patience "$patience" \
                    -lr "$lr" \
                    -criterion ce \
                    -gpuID 0 \
                    -force_training \
                    -output_dir './results_RQ2/EmpAcc/adv' \
                    -model_root './models_adv' \
                    -attack_n_inject_max "$attackNnode" \
                    -attack_n_edge_max "$attackNedge" \
                    -attack_lr "$attacklr" \
                    -attack_epochs "$attackepc" \
                    -attack_sequential_step 0.2 \
                    -attack_quiet

                # Vanilla GNN
                python main_attack_vanilla.py \
                    -dataset "$dataset" \
                    -model "$model" \
                    -attack_method "$attack_method" \
                    -epochs "$EPOCH" \
                    -eval_interval 5 \
                    -patience "$patience" \
                    -lr "$lr" \
                    -criterion ce \
                    -gpuID 0 \
                    -force_training \
                    -output_dir './results_RQ2/EmpAcc/vanilla' \
                    -model_root './models_vanilla' \
                    -attack_n_inject_max "$attackNnode" \
                    -attack_n_edge_max "$attackNedge" \
                    -attack_lr "$attacklr" \
                    -attack_epochs "$attackepc" \
                    -attack_sequential_step 0.2 \
                    -attack_quiet
            done
        done
    done
done
