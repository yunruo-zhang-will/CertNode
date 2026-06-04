import math

import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax


def _ensure_long_edge_index(edge_index, device):
    return edge_index.to(device=device, dtype=torch.long)


def _add_self_loops(edge_index, edge_weight, num_nodes, device, fill_value=1.0):
    loop_index = torch.arange(num_nodes, device=device, dtype=torch.long)
    loop_edges = torch.stack([loop_index, loop_index], dim=0)
    loop_weight = torch.full((num_nodes,), float(fill_value), device=device, dtype=edge_weight.dtype)
    return torch.cat([edge_index, loop_edges], dim=1), torch.cat([edge_weight, loop_weight], dim=0)


def _normalize_incoming(edge_weight, dst, num_nodes):
    denom = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    denom.index_add_(0, dst, edge_weight)
    denom = denom.clamp_min(1e-12)
    return edge_weight / denom[dst]


def _weighted_sum(messages, dst, num_nodes):
    out = torch.zeros(num_nodes, messages.size(-1), device=messages.device, dtype=messages.dtype)
    out.index_add_(0, dst, messages)
    return out


def _guard_attention(x, edge_index, threshold, temperature, previous_attention=None, memory_alpha=0.0):
    src, dst = edge_index
    sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
    sim = ((sim + 1.0) * 0.5).clamp_min(0.0)
    if temperature != 1.0:
        sim = sim.pow(1.0 / max(temperature, 1e-6))
    sim = torch.where(sim >= threshold, sim, torch.zeros_like(sim))
    attention = _normalize_incoming(sim, dst, x.size(0))
    if previous_attention is not None and previous_attention.shape == attention.shape:
        attention = memory_alpha * previous_attention + (1.0 - memory_alpha) * attention
        attention = _normalize_incoming(attention, dst, x.size(0))
    return attention


class GuardConfigMixin:
    def __init__(self, guard_threshold=0.1, guard_temperature=1.0, guard_memory=0.5):
        self.guard_threshold = guard_threshold
        self.guard_temperature = guard_temperature
        self.guard_memory = guard_memory

    def _compute_guard_edges(self, x, edge_index, previous_attention=None, add_self_loops=False):
        edge_index = _ensure_long_edge_index(edge_index, x.device)
        edge_weight = _guard_attention(
            x,
            edge_index,
            threshold=self.guard_threshold,
            temperature=self.guard_temperature,
            previous_attention=previous_attention,
            memory_alpha=self.guard_memory,
        )
        guarded_edge_index = edge_index
        guarded_edge_weight = edge_weight
        if add_self_loops:
            guarded_edge_index, guarded_edge_weight = _add_self_loops(
                edge_index,
                edge_weight,
                x.size(0),
                x.device,
            )
            guarded_edge_weight = _normalize_incoming(guarded_edge_weight, guarded_edge_index[1], x.size(0))
        return guarded_edge_index, guarded_edge_weight, edge_weight


class GuardedGCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        transformed = x @ self.weight
        messages = transformed[src] * edge_weight.unsqueeze(-1)
        return _weighted_sum(messages, dst, x.size(0)) + self.bias


class GuardedSAGELayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        self.lin_neigh = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        neigh = _weighted_sum(x[src] * edge_weight.unsqueeze(-1), dst, x.size(0))
        return self.lin_self(x) + self.lin_neigh(neigh)


class GuardedGINLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index
        neigh = _weighted_sum(x[src] * edge_weight.unsqueeze(-1), dst, x.size(0))
        return self.mlp((1.0 + self.eps) * x + neigh)


class GuardedGATLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = torch.nn.Parameter(torch.empty(out_channels))
        self.att_dst = torch.nn.Parameter(torch.empty(out_channels))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        bound = 1.0 / math.sqrt(self.att_src.numel())
        torch.nn.init.uniform_(self.att_src, -bound, bound)
        torch.nn.init.uniform_(self.att_dst, -bound, bound)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, guard_weight):
        src, dst = edge_index
        transformed = self.lin(x)
        alpha_src = (transformed[src] * self.att_src).sum(dim=-1)
        alpha_dst = (transformed[dst] * self.att_dst).sum(dim=-1)
        logits = F.leaky_relu(alpha_src + alpha_dst, negative_slope=self.negative_slope)

        alpha = softmax(logits + torch.log(guard_weight.clamp_min(1e-12)), dst, num_nodes=x.size(0))
        out = _weighted_sum(transformed[src] * alpha.unsqueeze(-1), dst, x.size(0))
        return out + self.bias


class NodeGCNGNNGuard(torch.nn.Module, GuardConfigMixin):
    def __init__(self, num_features, num_classes, hidden_size=64, guard_threshold=0.1, guard_temperature=1.0, guard_memory=0.5):
        torch.nn.Module.__init__(self)
        GuardConfigMixin.__init__(self, guard_threshold, guard_temperature, guard_memory)
        self.embedding_size = hidden_size * 3
        self.conv1 = GuardedGCNLayer(num_features, hidden_size)
        self.conv2 = GuardedGCNLayer(hidden_size, hidden_size)
        self.conv3 = GuardedGCNLayer(hidden_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size * 3, num_classes)

    def embedding(self, x, edge_index):
        stack = []
        attention = None

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
            x,
            edge_index,
            attention,
            add_self_loops=True,
        )
        out1 = self.conv1(x, guarded_edge_index, guarded_attention)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)
        stack.append(out1)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
            out1,
            edge_index,
            attention,
            add_self_loops=True,
        )
        out2 = self.conv2(out1, guarded_edge_index, guarded_attention)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)
        stack.append(out2)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
            out2,
            edge_index,
            attention,
            add_self_loops=True,
        )
        out3 = self.conv3(out2, guarded_edge_index, guarded_attention)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)
        stack.append(out3)
        return torch.cat(stack, dim=1)

    def forward(self, x, edge_index):
        return self.lin(self.embedding(x, edge_index))


class NodeGATGNNGuard(torch.nn.Module, GuardConfigMixin):
    def __init__(self, num_features, num_classes, hidden_size=64, guard_threshold=0.1, guard_temperature=1.0, guard_memory=0.5):
        torch.nn.Module.__init__(self)
        GuardConfigMixin.__init__(self, guard_threshold, guard_temperature, guard_memory)
        self.embedding_size = hidden_size * 3
        self.conv1 = GuardedGATLayer(num_features, hidden_size)
        self.conv2 = GuardedGATLayer(hidden_size, hidden_size)
        self.conv3 = GuardedGATLayer(hidden_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size * 3, num_classes)

    def embedding(self, x, edge_index):
        stack = []
        attention = None

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
            x,
            edge_index,
            attention,
            add_self_loops=True,
        )
        out1 = self.conv1(x, guarded_edge_index, guarded_attention)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)
        stack.append(out1)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
            out1,
            edge_index,
            attention,
            add_self_loops=True,
        )
        out2 = self.conv2(out1, guarded_edge_index, guarded_attention)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)
        stack.append(out2)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
            out2,
            edge_index,
            attention,
            add_self_loops=True,
        )
        out3 = self.conv3(out2, guarded_edge_index, guarded_attention)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)
        stack.append(out3)
        return torch.cat(stack, dim=1)

    def forward(self, x, edge_index):
        return self.lin(self.embedding(x, edge_index))


class NodeGSAGEGNNGuard(torch.nn.Module, GuardConfigMixin):
    def __init__(self, num_features, num_classes, hidden_size=64, guard_threshold=0.1, guard_temperature=1.0, guard_memory=0.5):
        torch.nn.Module.__init__(self)
        GuardConfigMixin.__init__(self, guard_threshold, guard_temperature, guard_memory)
        self.embedding_size = hidden_size * 3
        self.conv1 = GuardedSAGELayer(num_features, hidden_size)
        self.conv2 = GuardedSAGELayer(hidden_size, hidden_size)
        self.conv3 = GuardedSAGELayer(hidden_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size * 3, num_classes)

    def embedding(self, x, edge_index):
        stack = []
        attention = None

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(x, edge_index, attention)
        out1 = self.conv1(x, guarded_edge_index, guarded_attention)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)
        stack.append(out1)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(out1, edge_index, attention)
        out2 = self.conv2(out1, guarded_edge_index, guarded_attention)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)
        stack.append(out2)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(out2, edge_index, attention)
        out3 = self.conv3(out2, guarded_edge_index, guarded_attention)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)
        stack.append(out3)
        return torch.cat(stack, dim=1)

    def forward(self, x, edge_index):
        return self.lin(self.embedding(x, edge_index))


class NodeGINGNNGuard(torch.nn.Module, GuardConfigMixin):
    def __init__(self, num_features, num_classes, hidden_size=64, guard_threshold=0.1, guard_temperature=1.0, guard_memory=0.5):
        torch.nn.Module.__init__(self)
        GuardConfigMixin.__init__(self, guard_threshold, guard_temperature, guard_memory)
        self.embedding_size = hidden_size * 3
        self.conv1 = GuardedGINLayer(num_features, hidden_size)
        self.conv2 = GuardedGINLayer(hidden_size, hidden_size)
        self.conv3 = GuardedGINLayer(hidden_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size * 3, num_classes)

    def embedding(self, x, edge_index):
        stack = []
        attention = None

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(x, edge_index, attention)
        out1 = self.conv1(x, guarded_edge_index, guarded_attention)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)
        stack.append(out1)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(out1, edge_index, attention)
        out2 = self.conv2(out1, guarded_edge_index, guarded_attention)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)
        stack.append(out2)

        guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(out2, edge_index, attention)
        out3 = self.conv3(out2, guarded_edge_index, guarded_attention)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)
        stack.append(out3)
        return torch.cat(stack, dim=1)

    def forward(self, x, edge_index):
        return self.lin(self.embedding(x, edge_index))


class NodeAPPNPGNNGuard(torch.nn.Module, GuardConfigMixin):
    def __init__(self, num_features, num_classes, hidden_size=64, K=10, alpha=0.1, guard_threshold=0.1, guard_temperature=1.0, guard_memory=0.5):
        torch.nn.Module.__init__(self)
        GuardConfigMixin.__init__(self, guard_threshold, guard_temperature, guard_memory)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )
        self.K = K
        self.alpha = alpha

    def forward(self, x, edge_index):
        logits = self.mlp(x)
        initial_logits = logits
        attention = None
        for _ in range(self.K):
            guarded_edge_index, guarded_attention, attention = self._compute_guard_edges(
                logits,
                edge_index,
                attention,
                add_self_loops=True,
            )
            src, dst = guarded_edge_index
            propagated = _weighted_sum(logits[src] * guarded_attention.unsqueeze(-1), dst, logits.size(0))
            logits = (1.0 - self.alpha) * propagated + self.alpha * initial_logits
        return logits