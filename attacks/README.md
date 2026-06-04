# Node Injection Attack Usage

This directory contains EOT-style node injection attacks adapted for CertNode.

The implementations are in [attacks/tdgia_eot.py](attacks/tdgia_eot.py), [attacks/gnia_eot.py](attacks/gnia_eot.py), and [attacks/grb_injection_baseline.py](attacks/grb_injection_baseline.py). The CLI entry point is in [main.py](main.py).

## What It Does

These attacks are designed for the smoothed node-classification models in CertNode.

Instead of attacking a single forward pass, they attack the smoothing distribution itself:

1. It repeatedly samples perturbed graphs using the same node-deletion mechanism as `forward_perturb`.
2. It averages the predicted probabilities over those samples.
3. It optimizes injected node topology and injected node features against that expectation.
4. Final evaluation still uses the real majority-vote smoothing procedure in `smoothed_precit`.

This makes the attack target consistent with the actual smoothed inference path.

Available methods:

1. `tdgia`: sequential topology injection plus gradient-based feature optimization.
2. `gnia`: a G-NIA-style local subgraph injection attack with homophily-aware feature regularization.
3. `grb_injection`: a GRB-style simple injection baseline with global vulnerable-target attachment and sign-gradient feature updates.

## Environment

Use the `gnn2` environment.

Example:

```bash
conda run -n gnn2 python main.py \
  -task Node \
  -dataset CiteSeer \
  -model GCN \
  -p_n 0.9 \
  -n_smoothing 10000 \
  -run_attack \
  -attack_method tdgia \
  -attack_n_inject_max 20 \
  -attack_n_edge_max 5 \
  -attack_epochs 100 \
  -attack_rank_samples 32 \
  -attack_opt_samples 16 \
  -attack_clean_smoothing 10000 \
  -attack_eval_smoothing 10000
```

## Recommended First Run

For the first experiment, start with:

1. `-task Node`
2. `-dataset CiteSeer`
3. `-model GCN`
4. `-attack_n_inject_max 20`
5. `-attack_n_edge_max 5`
6. `-attack_rank_samples 32`
7. `-attack_opt_samples 16`
8. `-attack_clean_smoothing 10000`
9. `-attack_eval_smoothing 10000`

This keeps the attack cheap enough to debug while still being meaningful.

## CLI Arguments

The following attack-specific arguments are added in [main.py](main.py):

1. `-run_attack`
Enables the selected node injection attack after clean smoothing evaluation.

2. `-attack_method`
Chooses one of `tdgia`, `gnia`, or `grb_injection`.

3. `-attack_n_inject_max`
Maximum number of injected nodes.

4. `-attack_n_edge_max`
Maximum number of edges created for each injected node.

5. `-attack_lr`
Learning rate for injected feature optimization.

6. `-attack_epochs`
Number of optimization steps used to update injected features.

7. `-attack_rank_samples`
Number of smoothing samples used when ranking target nodes for topology injection.

8. `-attack_opt_samples`
Number of smoothing samples used inside each optimization step of injected features.

9. `-attack_gnia_homophily`
Regularization weight used only by the G-NIA-style attack.

10. `-attack_clean_smoothing`
Number of smoothing samples used to compute clean majority-vote labels for attack target selection.

11. `-attack_eval_smoothing`
Number of smoothing samples used for attacked-graph evaluation.

12. `-attack_sequential_step`
Fraction of injected nodes added at each sequential attack step.

13. `-attack_quiet`
Suppresses per-step attack progress logs.

## Attack Flow

When `-run_attack` is enabled, [main.py](main.py) does the following for node classification:

1. Runs clean smoothing using `model.smoothed_precit(...)`.
2. Uses `attack_clean_smoothing` samples to obtain the clean majority-vote top-1 label used by the attack.
3. Restricts the attack target set to test nodes that are clean-correct under smoothing.
4. Runs the selected node injection attack.
5. Re-evaluates the attacked graph using the real `smoothed_precit(...)` function with `attack_eval_smoothing` samples.
6. Saves attacked metrics and the attacked graph.

## ASR Definition

Attack success rate is defined on clean-correct test nodes only.

Let:

1. `clean_top1` be the majority-vote prediction on the clean graph.
2. `attacked_top1` be the majority-vote prediction on the attacked graph.

Then:

```text
ASR = (# test nodes that are clean-correct and attacked-wrong)
      / (# test nodes that are clean-correct)
```

This avoids inflating attack success with nodes that were already misclassified on the clean graph.

## Output Files

When attack is enabled, the following extra files are written into the normal result directory created by [main.py](main.py):

1. `attack_result.csv`
Attack metrics summary in CSV format, including final accuracy and ASR, attack budgets, smoothing/sample settings, sequential attack rounds, optimization epochs, and theoretical model call count.

2. `attacked_graph.pt`
Serialized attacked graph payload, including:

```text
features
edge_index
injected_nodes
target_index
attacked_top1
attacked_top2
attacked_count1
attacked_count2
```

## Current Limitations

1. Only the `Node` task is supported.
2. The attack is implemented for smoothed models that expose the same perturbation logic as the current CertNode node models.
3. The attacked-graph certification curve is not yet recomputed separately; current attack evaluation focuses on attacked accuracy and ASR.
4. Large datasets such as Amazon may be much slower because smoothing and EOT both require repeated sampling.

## Notes About `__pycache__`

The `__pycache__` directory is not required source code.

It is auto-generated by Python when modules are imported. It is safe to delete. If you run the attack again, Python may recreate it automatically.