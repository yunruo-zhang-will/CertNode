import torch
import torch.nn.functional as F

from attacks.injection_common import (
    AttackMeta,
    append_undirected_edges,
    build_attack_info,
    eot_margin_loss,
    smoothed_predict_proba_grad,
    target_margin_scores,
)


class EOTGNIAAttack:
    def __init__(
        self,
        n_inject_max,
        n_edge_max,
        feat_lim_min,
        feat_lim_max,
        lr=0.01,
        n_epoch=100,
        sequential_step=0.2,
        n_rank_samples=16,
        n_opt_samples=8,
        homophily_weight=0.5,
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
        self.homophily_weight = homophily_weight
        self.device = device
        self.verbose = verbose
        self.last_attack_info = None

    def _rank_targets(self, mean_probs, anchor_labels, target_mask):
        target_index = torch.where(target_mask)[0]
        if target_index.numel() == 0:
            return target_index, torch.empty(0, device=mean_probs.device)
        margin = target_margin_scores(mean_probs, anchor_labels, target_index)
        return target_index, -margin

    def _neighbors_of(self, edge_index, node_id, n_original):
        src, dst = edge_index
        mask = (src == node_id) & (dst < n_original)
        return dst[mask].unique()

    def _inject_topology(self, features, edge_index, mean_probs, anchor_labels, target_mask, n_step, n_original):
        device = edge_index.device
        target_index, score = self._rank_targets(mean_probs[:n_original], anchor_labels, target_mask)
        injected_nodes = torch.arange(features.shape[0], features.shape[0] + n_step, device=device)
        if target_index.numel() == 0:
            init_features = torch.empty(n_step, features.shape[1], device=device).uniform_(self.feat_lim_min, self.feat_lim_max)
            return edge_index, injected_nodes, init_features

        order = torch.argsort(score, descending=True)
        ranked_targets = target_index[order]
        prototypes = []
        new_edges = []

        for offset, new_node in enumerate(injected_nodes.tolist()):
            seed = ranked_targets[offset % ranked_targets.numel()]
            neighbors = self._neighbors_of(edge_index, int(seed.item()), n_original)
            seed_feature = features[seed]
            picked_neighbors = []
            if neighbors.numel() > 0 and self.n_edge_max > 1:
                neighbor_features = features[neighbors]
                similarity = F.cosine_similarity(
                    neighbor_features,
                    seed_feature.unsqueeze(0).expand_as(neighbor_features),
                    dim=1,
                )
                topk = min(neighbors.numel(), self.n_edge_max - 1)
                picked_neighbors = neighbors[torch.topk(similarity, k=topk, largest=True).indices].tolist()

            connect_nodes = [int(seed.item())] + [int(node_id) for node_id in picked_neighbors]
            for target_node in connect_nodes[:self.n_edge_max]:
                new_edges.append([new_node, target_node])
                new_edges.append([target_node, new_node])

            proto_nodes = connect_nodes if connect_nodes else [int(seed.item())]
            prototype = features[torch.tensor(proto_nodes, device=device)].mean(dim=0)
            prototypes.append(prototype)

        init_features = torch.stack(prototypes, dim=0)
        init_features = init_features + 0.01 * torch.randn_like(init_features)
        init_features = init_features.clamp(self.feat_lim_min, self.feat_lim_max)
        return append_undirected_edges(edge_index, new_edges, device), injected_nodes, init_features

    def _optimize_features(self, model, features_attack, edge_index_attack, anchor_labels, target_mask, prototypes, no_dense=False):
        if self.n_epoch <= 0 or target_mask.sum().item() == 0:
            return features_attack, 0

        n_original = anchor_labels.shape[0]
        base_features = features_attack[:n_original].detach()
        injected_features = features_attack[n_original:].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([injected_features], lr=self.lr)
        target_index = torch.where(target_mask)[0]
        prototype_count = prototypes.shape[0]

        model.eval()
        executed_epochs = 0
        for _ in range(self.n_epoch):
            full_features = torch.cat([base_features, injected_features], dim=0)
            mean_probs = smoothed_predict_proba_grad(
                model,
                full_features,
                edge_index_attack,
                num_samples=self.n_opt_samples,
                no_dense=no_dense,
            )
            attack_loss = eot_margin_loss(mean_probs[:n_original], anchor_labels, target_index)
            # Only the newly injected nodes in the current round have matching prototypes.
            current_injected = injected_features[-prototype_count:] if prototype_count > 0 else injected_features
            homophily_loss = ((current_injected - prototypes) ** 2).mean() if prototype_count > 0 else attack_loss * 0.0
            loss = attack_loss + self.homophily_weight * homophily_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                injected_features.clamp_(self.feat_lim_min, self.feat_lim_max)
            executed_epochs += 1

        return torch.cat([base_features, injected_features.detach()], dim=0), executed_epochs

    def attack(self, model, features, edge_index, anchor_labels, target_mask, no_dense=False):
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
            mean_probs = smoothed_predict_proba_grad(
                model,
                current_features,
                current_edge_index,
                num_samples=self.n_rank_samples,
                no_dense=no_dense,
            )
            current_edge_index, injected_nodes, init_features = self._inject_topology(
                current_features,
                current_edge_index,
                mean_probs,
                anchor_labels,
                target_mask,
                n_step,
                n_original,
            )
            current_features = torch.cat([current_features, init_features], dim=0)
            current_features, executed_epochs = self._optimize_features(
                model,
                current_features,
                current_edge_index,
                anchor_labels,
                target_mask,
                prototypes=init_features.detach(),
                no_dense=no_dense,
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
            n_rank_samples=self.n_rank_samples,
            n_opt_samples=self.n_opt_samples,
        )
        return current_features, current_edge_index, AttackMeta(
            n_original=n_original,
            injected_nodes=injected_nodes,
            target_index=torch.where(target_mask)[0],
            target_count=int(target_mask.sum().item()),
        ), self.last_attack_info