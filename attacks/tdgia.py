import math

import torch

from attacks.injection_common import (
    AttackMeta,
    append_undirected_edges,
    build_attack_info,
    edge_degree,
    eot_margin_loss,
    standard_predict_proba_grad,
)


class TDGIAAttack:
    def __init__(
        self,
        n_inject_max,
        n_edge_max,
        feat_lim_min,
        feat_lim_max,
        lr=0.01,
        n_epoch=100,
        sequential_step=0.2,
        weight1=0.9,
        weight2=0.1,
        n_rank_samples=1,
        n_opt_samples=1,
        device="cpu",
        verbose=True,
    ):
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.lr = lr
        self.n_epoch = n_epoch
        self.sequential_step = sequential_step
        self.weight1 = weight1
        self.weight2 = weight2
        self.n_rank_samples = n_rank_samples
        self.n_opt_samples = n_opt_samples
        self.device = device
        self.verbose = verbose
        self.last_attack_info = None

    def _rank_targets(self, edge_index, mean_probs, anchor_labels, target_mask, current_num_nodes):
        target_index = torch.where(target_mask)[0]
        if target_index.numel() == 0:
            return target_index, torch.empty(0, device=edge_index.device)
        degree = edge_degree(edge_index, current_num_nodes, edge_index.device)
        chosen_prob = mean_probs[target_index, anchor_labels[target_index]] + 2.0
        score1 = chosen_prob / degree[target_index]
        score2 = chosen_prob / torch.sqrt(degree[target_index])
        score = self.weight1 * score1 + self.weight2 * score2 / math.sqrt(max(1, self.n_edge_max))
        return target_index, score

    def _inject_topology(self, edge_index, mean_probs, anchor_labels, target_mask, n_inject_cur, current_num_nodes):
        device = edge_index.device
        current_nodes = current_num_nodes
        target_index, score = self._rank_targets(edge_index, mean_probs, anchor_labels, target_mask, current_num_nodes)
        if target_index.numel() == 0:
            injected_nodes = torch.arange(current_nodes, current_nodes + n_inject_cur, device=device)
            return edge_index, injected_nodes

        k = min(target_index.numel(), n_inject_cur * self.n_edge_max)
        picked = target_index[torch.topk(score, k=k, largest=True).indices]

        grouped = {}
        for node_id in picked.tolist():
            label = int(anchor_labels[node_id].item())
            grouped.setdefault(label, []).append(node_id)

        class_ids = list(grouped.keys())
        class_pos = {class_id: 0 for class_id in class_ids}
        injected_nodes = torch.arange(current_nodes, current_nodes + n_inject_cur, device=device)
        new_edges = []

        for new_node in injected_nodes.tolist():
            for _ in range(self.n_edge_max):
                best_class = None
                best_ratio = float("inf")
                for class_id in class_ids:
                    nodes = grouped[class_id]
                    if class_pos[class_id] < len(nodes):
                        ratio = class_pos[class_id] / max(1, len(nodes))
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_class = class_id
                if best_class is None:
                    break
                target_node = grouped[best_class][class_pos[best_class]]
                class_pos[best_class] += 1
                new_edges.append([new_node, target_node])
                new_edges.append([target_node, new_node])

        return append_undirected_edges(edge_index, new_edges, device), injected_nodes

    def _optimize_features(self, model, features_attack, edge_index_attack, anchor_labels, target_mask, no_dense=False):
        if self.n_epoch <= 0 or target_mask.sum().item() == 0:
            return features_attack, 0

        del no_dense
        n_original = anchor_labels.shape[0]
        base_features = features_attack[:n_original].detach()
        injected_features = features_attack[n_original:].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([injected_features], lr=self.lr)
        target_index = torch.where(target_mask)[0]

        model.eval()
        executed_epochs = 0
        for _ in range(self.n_epoch):
            full_features = torch.cat([base_features, injected_features], dim=0)
            mean_probs = standard_predict_proba_grad(model, full_features, edge_index_attack)
            loss = eot_margin_loss(mean_probs[:n_original], anchor_labels, target_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                injected_features.clamp_(self.feat_lim_min, self.feat_lim_max)
            executed_epochs += 1

        return torch.cat([base_features, injected_features.detach()], dim=0), executed_epochs

    def attack(self, model, features, edge_index, anchor_labels, target_mask, no_dense=False):
        del no_dense
        model.eval()
        model.to(self.device)

        current_features = features.detach().clone().to(self.device)
        current_edge_index = edge_index.detach().clone().to(self.device)
        anchor_labels = anchor_labels.detach().clone().to(self.device)
        target_mask = target_mask.detach().clone().to(self.device)

        n_original = current_features.shape[0]
        feat_dim = current_features.shape[1]
        injected_all = []
        injected_so_far = 0
        ranking_rounds = 0
        optimization_epochs = 0

        while injected_so_far < self.n_inject_max:
            ranking_rounds += 1
            n_step = min(
                self.n_inject_max - injected_so_far,
                max(1, int(self.n_inject_max * self.sequential_step)),
            )
            mean_probs = standard_predict_proba_grad(model, current_features, current_edge_index)
            current_edge_index, injected_nodes = self._inject_topology(
                current_edge_index,
                mean_probs[:n_original],
                anchor_labels,
                target_mask,
                n_step,
                current_features.shape[0],
            )
            injected_all.append(injected_nodes)

            init_features = torch.empty(n_step, feat_dim, device=self.device).uniform_(self.feat_lim_min, self.feat_lim_max)
            current_features = torch.cat([current_features, init_features], dim=0)
            current_features, executed_epochs = self._optimize_features(
                model,
                current_features,
                current_edge_index,
                anchor_labels,
                target_mask,
            )
            optimization_epochs += executed_epochs
            injected_so_far += n_step
            if self.verbose:
                print(f"Injected {injected_so_far}/{self.n_inject_max} nodes")

        if injected_all:
            injected_nodes = torch.cat(injected_all, dim=0)
        else:
            injected_nodes = torch.empty(0, dtype=torch.long, device=self.device)

        self.last_attack_info = build_attack_info(
            ranking_rounds=ranking_rounds,
            optimization_epochs=optimization_epochs,
            sequential_rounds=ranking_rounds,
            n_rank_samples=1,
            n_opt_samples=1,
        )

        return current_features, current_edge_index, AttackMeta(
            n_original=n_original,
            injected_nodes=injected_nodes,
            target_index=torch.where(target_mask)[0],
            target_count=int(target_mask.sum().item()),
        ), self.last_attack_info