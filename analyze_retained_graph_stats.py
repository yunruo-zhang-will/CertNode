import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data

from dataset_loader import load_graph_data, load_node_data
from utils import remove_node_incident_edges_pyg_batch, remove_nodes_pyg


def init_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_p_values(raw: str) -> List[float]:
    values = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0 or value > 1:
            raise ValueError(f'p must be in [0, 1], got {value}')
        values.append(value)
    if not values:
        raise ValueError('at least one p value is required')
    return values


def load_amazon_products() -> Tuple[Data, int, int]:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise ImportError('dataset Amazon requires ogb to load ogbn-products') from exc

    dataset = PygNodePropPredDataset(name='ogbn-products')
    graph = dataset[0]
    data = Data(
        x=torch.as_tensor(graph.x, dtype=torch.float32),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
        y=torch.as_tensor(graph.y).view(-1),
    )
    num_features = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    return data, num_features, num_classes


def load_node_graph(args: argparse.Namespace) -> Tuple[Data, int, int]:
    if args.dataset == 'Amazon':
        return load_amazon_products()

    if args.dataset == 'PubMed':
        num_train = 2000
        num_val = 600
    elif args.dataset == 'computers':
        num_train = 400
        num_val = 133
    else:
        num_train = 150
        num_val = 50

    return load_node_data(args.dataset, num_train=num_train, num_val=num_val)


def select_graph_split(mask_split: List[torch.Tensor], split: str) -> torch.Tensor:
    if split == 'train':
        return mask_split[0]
    if split == 'val':
        return mask_split[1]
    if split == 'test':
        return mask_split[2]
    if split == 'all':
        total = mask_split[0].shape[0]
        return torch.ones(total, dtype=torch.bool)
    raise ValueError(f'unknown split: {split}')


def iter_graph_samples(args: argparse.Namespace) -> Iterable[Tuple[str, Data]]:
    if args.task == 'Node':
        data, _, _ = load_node_graph(args)
        yield args.dataset, data.cpu()
        return

    graphs, _, _, mask_split, _ = load_graph_data(args.dataset)
    selected_mask = select_graph_split(mask_split, args.graph_split)
    selected_indices = torch.where(selected_mask)[0].tolist()
    for graph_idx in selected_indices:
        yield f'{args.dataset}_{graph_idx}', graphs[graph_idx].cpu()


def unique_undirected_edges(edge_index: torch.Tensor) -> np.ndarray:
    if edge_index.numel() == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = edge_index.t().cpu().numpy().astype(np.int64, copy=False)
    no_self_loops = edges[:, 0] != edges[:, 1]
    edges = edges[no_self_loops]
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def sample_node_subgraph(data: Data, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index, keep_mask = remove_nodes_pyg(
        num_nodes=data.num_nodes,
        edge_index=data.edge_index,
        p=p,
        return_mask=True,
    )
    return edge_index.cpu(), keep_mask.cpu()


def sample_graph_subgraph(data: Data, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    edge_index, keep_mask = remove_node_incident_edges_pyg_batch(
        data.edge_index,
        batch,
        p,
        return_mask=True,
    )
    return edge_index.cpu(), keep_mask.cpu()


def compute_graph_metrics(original_edge_index: torch.Tensor, retained_edge_index: torch.Tensor, keep_mask: torch.Tensor) -> Dict[str, float]:
    retained_nodes = int(keep_mask.sum().item())
    original_edges = unique_undirected_edges(original_edge_index)
    retained_edges = unique_undirected_edges(retained_edge_index)
    original_edge_count = int(original_edges.shape[0])
    retained_edge_count = int(retained_edges.shape[0])

    if retained_nodes == 0:
        raise ValueError('retained_nodes should be positive')

    kept_nodes = torch.where(keep_mask)[0].cpu().numpy().astype(np.int64, copy=False)
    node_to_local = {node: idx for idx, node in enumerate(kept_nodes.tolist())}
    degree = np.zeros(retained_nodes, dtype=np.int64)

    if retained_edge_count > 0:
        rows = np.fromiter((node_to_local[int(src)] for src in retained_edges[:, 0]), dtype=np.int64)
        cols = np.fromiter((node_to_local[int(dst)] for dst in retained_edges[:, 1]), dtype=np.int64)
        degree += np.bincount(rows, minlength=retained_nodes)
        degree += np.bincount(cols, minlength=retained_nodes)
        graph_adj = sp.coo_matrix(
            (
                np.ones(rows.shape[0] * 2, dtype=np.int8),
                (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
            ),
            shape=(retained_nodes, retained_nodes),
        ).tocsr()
        _, component_labels = connected_components(graph_adj, directed=False, return_labels=True)
        largest_component_size = int(np.bincount(component_labels).max())
    else:
        largest_component_size = 1

    isolated_kept_nodes = int((degree == 0).sum())
    retained_edge_ratio = 0.0 if original_edge_count == 0 else retained_edge_count / original_edge_count

    return {
        'retained_node_count': retained_nodes,
        'retained_node_ratio': retained_nodes / int(keep_mask.numel()),
        'retained_edge_count': retained_edge_count,
        'retained_edge_ratio': retained_edge_ratio,
        'isolated_node_count': isolated_kept_nodes,
        'isolated_node_ratio': isolated_kept_nodes / retained_nodes,
        'largest_component_ratio': largest_component_size / retained_nodes,
        'average_degree': (2.0 * retained_edge_count) / retained_nodes,
    }


def summarize_results(rows: List[Dict[str, float]], p_values: List[float]) -> List[Dict[str, float]]:
    metric_names = [
        'retained_node_ratio',
        'retained_edge_ratio',
        'isolated_node_ratio',
        'largest_component_ratio',
        'average_degree',
    ]
    summaries: List[Dict[str, float]] = []
    for p in p_values:
        subset = [row for row in rows if abs(row['p'] - p) < 1e-12]
        if not subset:
            continue
        summary: Dict[str, float] = {
            'p': p,
            'num_records': len(subset),
        }
        for metric in metric_names:
            values = np.array([row[metric] for row in subset], dtype=np.float64)
            summary[f'{metric}_mean'] = float(values.mean())
            summary[f'{metric}_std'] = float(values.std(ddof=0))
        summaries.append(summary)
    return summaries


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(args: argparse.Namespace) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    graph_items = list(iter_graph_samples(args))

    for sample_round in range(args.num_samples):
        for graph_name, data in graph_items:
            if args.task == 'Node':
                retained_edge_index, keep_mask = sample_node_subgraph(data, args.current_p)
            else:
                retained_edge_index, keep_mask = sample_graph_subgraph(data, args.current_p)

            metrics = compute_graph_metrics(data.edge_index.cpu(), retained_edge_index, keep_mask)
            rows.append({
                'p': args.current_p,
                'sample_id': sample_round,
                'graph_id': graph_name,
                **metrics,
            })

    return rows, summarize_results(rows, [args.current_p])


def main(args: argparse.Namespace) -> None:
    init_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    p_values = parse_p_values(args.p_values)

    all_rows: List[Dict[str, float]] = []
    all_summaries: List[Dict[str, float]] = []
    for p in p_values:
        args.current_p = p
        rows, summaries = analyze_dataset(args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        print(
            f'p={p:.4f} analyzed over {len(rows)} graph instances '
            f'({args.num_samples} samples x per-split graphs).'
        )

    detail_path = os.path.join(args.output_dir, 'retained_graph_stats_detail.csv')
    summary_path = os.path.join(args.output_dir, 'retained_graph_stats_summary.csv')
    write_csv(detail_path, all_rows)
    write_csv(summary_path, all_summaries)
    print(f'Saved detailed rows to {detail_path}')
    print(f'Saved summary rows to {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze retained-subgraph statistics under smoothing-style node deletion.')
    parser.add_argument('-task', type=str, default='Node', choices=['Node', 'Graph'])
    parser.add_argument(
        '-dataset',
        type=str,
        default='CiteSeer',
        choices=['PubMed', 'CiteSeer', 'computers', 'Cora-ML', 'Amazon', 'Mutagenicity', 'PROTEINS', 'DD', 'AIDS'],
    )
    parser.add_argument('-graph_split', type=str, default='test', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('-p_values', type=str, default='0.1,0.3,0.5,0.7,0.9')
    parser.add_argument('-num_samples', type=int, default=20, help='independent deletion samples per p value')
    parser.add_argument('-seed', type=int, default=2020)
    parser.add_argument('-output_dir', type=str, default='./results_graph_stats')

    main(parser.parse_args())