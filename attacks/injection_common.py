import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from utils import remove_nodes_pyg


@dataclass
class AttackMeta:
    n_original: int
    injected_nodes: torch.Tensor
    target_index: torch.Tensor
    target_count: int


@dataclass
class AttackEvaluation:
    clean_accuracy: float
    attacked_accuracy: float
    asr: float
    attacked_top1: np.ndarray
    attacked_top2: np.ndarray
    attacked_count1: list
    attacked_count2: list
    clean_correct_count: int
    eval_count: int


@dataclass
class AttackInfo:
    ranking_rounds: int
    optimization_epochs: int
    sequential_rounds: int
    n_rank_samples: int
    n_opt_samples: int
    theoretical_model_calls: int


def sample_perturbed_edge_index(model, features, edge_index, no_dense=False):
    num_nodes = features.shape[0]
    if no_dense:
        sampled = remove_nodes_pyg(num_nodes, edge_index, model.p_n)
        sampled_edge_index = sampled[0] if isinstance(sampled, tuple) else sampled
    else:
        adj_dense = torch.squeeze(to_dense_adj(edge_index, max_num_nodes=num_nodes))
        adj_dense = model.perturbation(adj_dense)
        sampled_edge_index = torch.nonzero(adj_dense, as_tuple=False).t().contiguous()
    return sampled_edge_index.to(edge_index.device)


def smoothed_predict_proba_grad(model, features, edge_index, num_samples, no_dense=False):
    probs_sum = None
    for _ in range(num_samples):
        sampled_edge_index = sample_perturbed_edge_index(model, features, edge_index, no_dense=no_dense)
        logits = model(features, sampled_edge_index)
        probs = F.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / float(num_samples)


def standard_predict_proba_grad(model, features, edge_index):
    logits = model(features, edge_index)
    return F.softmax(logits, dim=1)


def edge_degree(edge_index, num_nodes, device):
    degree = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))
    return degree.clamp_min(1.0)


def append_undirected_edges(edge_index, new_edges, device):
    if not new_edges:
        return edge_index
    extra = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
    return torch.cat([edge_index, extra], dim=1)


def eot_margin_loss(mean_probs, anchor_labels, target_index):
    if target_index.numel() == 0:
        return mean_probs.sum() * 0.0
    selected = mean_probs[target_index]
    chosen = anchor_labels[target_index]
    chosen_prob = selected.gather(1, chosen.view(-1, 1)).squeeze(1)
    other_prob = selected.clone()
    other_prob.scatter_(1, chosen.view(-1, 1), -1.0)
    runner_up = other_prob.max(dim=1).values
    return (chosen_prob - runner_up).mean()


def target_margin_scores(mean_probs, anchor_labels, target_index):
    if target_index.numel() == 0:
        return torch.empty(0, device=mean_probs.device)
    selected = mean_probs[target_index]
    chosen = anchor_labels[target_index]
    chosen_prob = selected.gather(1, chosen.view(-1, 1)).squeeze(1)
    other_prob = selected.clone()
    other_prob.scatter_(1, chosen.view(-1, 1), -1.0)
    runner_up = other_prob.max(dim=1).values
    return chosen_prob - runner_up


def build_attack_info(ranking_rounds, optimization_epochs, sequential_rounds, n_rank_samples, n_opt_samples):
    return AttackInfo(
        ranking_rounds=ranking_rounds,
        optimization_epochs=optimization_epochs,
        sequential_rounds=sequential_rounds,
        n_rank_samples=n_rank_samples,
        n_opt_samples=n_opt_samples,
        theoretical_model_calls=ranking_rounds * n_rank_samples + optimization_epochs * n_opt_samples,
    )


def evaluate_smoothed_attack(
    model,
    clean_features,
    clean_edge_index,
    attacked_features,
    attacked_edge_index,
    labels,
    test_mask,
    num_smoothing,
    no_dense=False,
    clean_top1=None,
):
    if clean_top1 is None:
        clean_top2, _, _ = model.smoothed_precit(clean_features, clean_edge_index, num=num_smoothing, no_dense=no_dense)
        clean_top1 = np.asarray(clean_top2[:, 0])
    else:
        clean_top1 = np.asarray(clean_top1)

    attacked_top2, attacked_count1, attacked_count2 = model.smoothed_precit(
        attacked_features,
        attacked_edge_index,
        num=num_smoothing,
        no_dense=no_dense,
    )
    attacked_top1 = np.asarray(attacked_top2[:, 0])

    if isinstance(test_mask, torch.Tensor):
        test_index = torch.where(test_mask.cpu())[0].numpy()
    else:
        test_index = np.where(np.asarray(test_mask))[0]

    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    clean_correct = clean_top1[test_index] == labels_np[test_index]
    attacked_correct = attacked_top1[test_index] == labels_np[test_index]
    attacked_wrong = ~attacked_correct

    clean_correct_count = int(clean_correct.sum())
    asr = 0.0
    if clean_correct_count > 0:
        asr = float(np.logical_and(clean_correct, attacked_wrong).sum() / clean_correct_count)

    return AttackEvaluation(
        clean_accuracy=float(clean_correct.mean()) if test_index.size > 0 else 0.0,
        attacked_accuracy=float(attacked_correct.mean()) if test_index.size > 0 else 0.0,
        asr=asr,
        attacked_top1=attacked_top1,
        attacked_top2=attacked_top2,
        attacked_count1=attacked_count1,
        attacked_count2=attacked_count2,
        clean_correct_count=clean_correct_count,
        eval_count=int(test_index.size),
    )


def evaluate_standard_attack(
    model,
    clean_features,
    clean_edge_index,
    attacked_features,
    attacked_edge_index,
    labels,
    test_mask,
    clean_top1=None,
):
    model.eval()

    with torch.no_grad():
        if clean_top1 is None:
            clean_logits = model(clean_features, clean_edge_index)
            clean_top1 = clean_logits.argmax(dim=1).detach().cpu().numpy()
        else:
            clean_top1 = np.asarray(clean_top1)

        attacked_logits = model(attacked_features, attacked_edge_index)
        attacked_probs = F.softmax(attacked_logits, dim=1)
        attacked_top2 = torch.topk(attacked_probs, k=2, dim=1).indices.detach().cpu().numpy()
        attacked_top1 = attacked_top2[:, 0]
        attacked_count1 = torch.topk(attacked_probs, k=2, dim=1).values[:, 0].detach().cpu().tolist()
        attacked_count2 = torch.topk(attacked_probs, k=2, dim=1).values[:, 1].detach().cpu().tolist()

    if isinstance(test_mask, torch.Tensor):
        if test_mask.dtype == torch.bool:
            test_index = torch.where(test_mask.cpu())[0].numpy()
        else:
            test_index = test_mask.detach().cpu().numpy().reshape(-1)
    else:
        test_mask = np.asarray(test_mask)
        test_index = np.where(test_mask)[0] if test_mask.dtype == bool else test_mask.reshape(-1)

    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    clean_correct = clean_top1[test_index] == labels_np[test_index]
    attacked_correct = attacked_top1[test_index] == labels_np[test_index]
    attacked_wrong = ~attacked_correct

    clean_correct_count = int(clean_correct.sum())
    asr = 0.0
    if clean_correct_count > 0:
        asr = float(np.logical_and(clean_correct, attacked_wrong).sum() / clean_correct_count)

    return AttackEvaluation(
        clean_accuracy=float(clean_correct.mean()) if test_index.size > 0 else 0.0,
        attacked_accuracy=float(attacked_correct.mean()) if test_index.size > 0 else 0.0,
        asr=asr,
        attacked_top1=attacked_top1,
        attacked_top2=attacked_top2,
        attacked_count1=attacked_count1,
        attacked_count2=attacked_count2,
        clean_correct_count=clean_correct_count,
        eval_count=int(test_index.size),
    )