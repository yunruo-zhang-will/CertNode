#!/usr/bin/env bash

set -euo pipefail

for dataset in CiteSeer PubMed computers Cora-ML
do
	echo "Running analysis for dataset=${dataset}"
	python analyze_retained_graph_stats.py \
		-task Node \
		-dataset "$dataset" \
		-p_values "0.4,0.99" \
		-num_samples 20 \
		-seed 2020 \
		-output_dir "./results_RQ4/p/graph_stats/${dataset}"
done

