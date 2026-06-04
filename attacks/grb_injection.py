import torch

from attacks.injection_common import (
    AttackMeta,
    append_undirected_edges,
    build_attack_info,
    eot_margin_loss,
    standard_predict_proba_grad,
    target_margin_scores,
)


class GRBInjectionAttack:
    def __init__(
        self,
        n_inject_max,
        n_edge_max,
        feat_lim_min,
        feat_lim_max,
        lr=0.01,
        n_epoch=100,
        sequential_step=0.2,
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
        self.n_rank_samples = n_rank_samples
        self.n_opt_samples = n_opt_samples
        self.device = device
        self.verbose = verbose
        self.last_attack_info = None

    def _rank_targets(self, mean_probs, anchor_labels, target_mask):
        target_index = torch.where(target_mask)[0]
        if target_index.numel() == 0:
            return target_index, torch.empty(0, device=mean_probs.device)
        margin = target_margin_scores(mean_probs, anchor_labels, target_index)
        return target_index, -margin

    def _inject_topology(self, features, edge_index, mean_probs, anchor_labels, target_mask, n_step):
        device = edge_index.device
        target_index, score = self._rank_targets(mean_probs, anchor_labels, target_mask)
        injected_nodes = torch.arange(features.shape[0], features.shape[0] + n_step, device=device)
        if target_index.numel() == 0:
            init_features = torch.empty(n_step, features.shape[1], device=device).uniform_(self.feat_lim_min, self.feat_lim_max)
            return edge_index, injected_nodes, init_features

        ranked_targets = target_index[torch.argsort(score, descending=True)]
        k = min(ranked_targets.numel(), max(1, n_step * self.n_edge_max))
        picked = ranked_targets[:k]
        new_edges = []
        for offset, new_node in enumerate(injected_nodes.tolist()):
            for edge_pos in range(self.n_edge_max):
                target_node = int(picked[(offset * self.n_edge_max + edge_pos) % picked.numel()].item())
                new_edges.append([new_node, target_node])
                new_edges.append([target_node, new_node])

        prototype = features[picked].mean(dim=0)
        init_features = prototype.unsqueeze(0).repeat(n_step, 1)
        return append_undirected_edges(edge_index, new_edges, device), injected_nodes, init_features

    def _optimize_features(self, model, features_attack, edge_index_attack, anchor_labels, target_mask, no_dense=False):
        if self.n_epoch <= 0 or target_mask.sum().item() == 0:
            return features_attack, 0

        del no_dense
        n_original = anchor_labels.shape[0]
        base_features = features_attack[:n_original].detach()
        injected_features = features_attack[n_original:].detach().clone().requires_grad_(True)
        target_index = torch.where(target_mask)[0]

        model.eval()
        executed_epochs = 0
        for _ in range(self.n_epoch):
            full_features = torch.cat([base_features, injected_features], dim=0)
            mean_probs = standard_predict_proba_grad(model, full_features, edge_index_attack)
            loss = eot_margin_loss(mean_probs[:n_original], anchor_labels, target_index)
            grad = torch.autograd.grad(loss, injected_features, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                injected_features -= self.lr * grad.sign()
                injected_features.clamp_(self.feat_lim_min, self.feat_lim_max)
            injected_features = injected_features.detach().requires_grad_(True)
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
            current_edge_index, injected_nodes, init_features = self._inject_topology(
                current_features,
                current_edge_index,
                mean_probs[:n_original],
                anchor_labels,
                target_mask,
                n_step,
            )
            current_features = torch.cat([current_features, init_features], dim=0)
            current_features, executed_epochs = self._optimize_features(
                model,
                current_features,
                current_edge_index,
                anchor_labels,
                target_mask,
            )
            injected_all.append(injected_nodes)
            optimization_epochs += executed_epochs
            injected_so_far += n_step
            if self.verbose:
                print(f"Injected {injected_so_far}/{self.n_inject_max} nodes")

        injected_nodes = torch.cat(injected_all, dim=0) if injected_all else torch.empty(0, dtype=torch.long, device=self.device)
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