import argparse
import csv
import math
import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from attacks.gnia import GNIAAttack
from attacks.gnia_eot import EOTGNIAAttack
from attacks.grb_injection import GRBInjectionAttack
from attacks.grb_injection_baseline import EOTGRBInjectionAttack
from attacks.injection_common import evaluate_smoothed_attack
from attacks.tdgia import TDGIAAttack
from attacks.tdgia_eot import EOTTDGIAAttack
from dataset_loader import load_node_data
from models import (
    SmoothNodeAPPNP,
    SmoothNodeGAT,
    SmoothNodeGCN,
    SmoothNodeGIN,
    SmoothNodeGSAGE,
)
from train import train_smoothing_model_nc


warnings.filterwarnings("ignore")


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean value, got: {value}')


def init_random_seed(seed=2021):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    warnings.filterwarnings("ignore")


def resolve_attack_seed(args):
    return args.seed if args.attack_seed is None else args.attack_seed


def set_attack_random_seed(seed):
    init_random_seed(seed)


def restart_seed(base_seed, restart_idx, offset=0):
    return int(base_seed + restart_idx + offset)


def format_attack_variant(method, use_eot):
    prefix = 'eot' if use_eot else 'base'
    return f'{prefix}_{method}'


def mask_to_index_tensor(mask, device):
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return torch.where(mask.to(device))[0]
        return mask.to(device).reshape(-1).long()
    return torch.as_tensor(mask, device=device, dtype=torch.long).reshape(-1)


def mask_to_bool_tensor(mask, num_nodes, device):
    if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
        return mask.to(device)
    mask_bool = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_bool[mask_to_index_tensor(mask, device)] = True
    return mask_bool


def load_node_dataset(args):
    if args.dataset == 'Amazon':
        from ogb.nodeproppred import PygNodePropPredDataset

        datasets = PygNodePropPredDataset(name="ogbn-products")
        graph = datasets[0]
        features = torch.as_tensor(graph.x, device=args.device)
        edge_index = torch.as_tensor(graph.edge_index, device=args.device)
        labels = torch.as_tensor(graph.y, device=args.device).squeeze()
        num_x = features.shape[1]
        num_labels = 47

        train_mask = []
        val_mask = []
        test_mask = []
        for class_id in range(num_labels):
            idx = (labels == class_id).nonzero(as_tuple=False).reshape(-1)
            train_i = idx[:int(0.3 * len(idx))]
            val_i = idx[int(0.3 * len(idx)):int(0.5 * len(idx))]
            test_i = idx[int(0.5 * len(idx)):]
            train_mask.extend(train_i.tolist())
            val_mask.extend(val_i.tolist())
            test_mask.extend(test_i.tolist())
    else:
        if args.dataset == 'PubMed':
            num_train = 2000
            num_val = 600
        elif args.dataset == 'computers':
            num_train = 400
            num_val = 133
        else:
            num_train = 150
            num_val = 50
        data, num_x, num_labels = load_node_data(args.dataset, num_train=num_train, num_val=num_val)
        features = torch.as_tensor(data.x, device=args.device)
        edge_index = torch.as_tensor(data.edge_index, device=args.device)
        labels = torch.as_tensor(data.y, device=args.device)
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    train_mask = mask_to_bool_tensor(train_mask, features.shape[0], args.device)
    val_mask = mask_to_bool_tensor(val_mask, features.shape[0], args.device)
    test_mask = mask_to_bool_tensor(test_mask, features.shape[0], args.device)
    return features, edge_index, labels, train_mask, val_mask, test_mask, num_x, num_labels


def build_smoothed_model(args, num_x, num_labels, sample_config):
    if args.model == 'GCN':
        return SmoothNodeGCN(num_x, num_labels, hidden_size=args.n_hidden, config=sample_config, device=args.device).to(args.device)
    if args.model == 'GAT':
        return SmoothNodeGAT(num_x, num_labels, hidden_size=args.n_hidden, config=sample_config, device=args.device).to(args.device)
    if args.model == 'GSAGE':
        return SmoothNodeGSAGE(num_x, num_labels, hidden_size=args.n_hidden, config=sample_config, device=args.device).to(args.device)
    if args.model == 'GIN':
        return SmoothNodeGIN(num_x, num_labels, hidden_size=args.n_hidden, config=sample_config, device=args.device).to(args.device)
    if args.model == 'APPNP':
        return SmoothNodeAPPNP(num_x, num_labels, hidden_size=args.n_hidden, config=sample_config, device=args.device).to(args.device)
    raise ValueError(f'unknown model: {args.model}')


def resolve_attack_sampling(args):
    if args.attack_use_eot:
        return args.attack_rank_samples, args.attack_opt_samples

    if (args.attack_rank_samples, args.attack_opt_samples) != (1, 1):
        print(
            'attack_use_eot=False: ranking and optimization use a single deterministic forward pass; '
            'effective attack_rank_samples=1 and attack_opt_samples=1.'
        )
    return 1, 1


def summarize_metric(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0
    return float(array.mean()), float(array.std(ddof=0))


def build_attack(args, features):
    effective_rank_samples, effective_opt_samples = resolve_attack_sampling(args)
    attack_kwargs = dict(
        n_inject_max=args.attack_n_inject_max,
        n_edge_max=args.attack_n_edge_max,
        feat_lim_min=float(features.min().item()),
        feat_lim_max=float(features.max().item()),
        lr=args.attack_lr,
        n_epoch=args.attack_epochs,
        sequential_step=args.attack_sequential_step,
        n_rank_samples=effective_rank_samples,
        n_opt_samples=effective_opt_samples,
        device=args.device,
        verbose=not args.attack_quiet,
    )
    if args.attack_method == 'tdgia':
        if args.attack_use_eot:
            return EOTTDGIAAttack(**attack_kwargs)
        return TDGIAAttack(**attack_kwargs)
    if args.attack_method == 'gnia':
        if args.attack_use_eot:
            return EOTGNIAAttack(**attack_kwargs, homophily_weight=args.attack_gnia_homophily)
        return GNIAAttack(**attack_kwargs, homophily_weight=args.attack_gnia_homophily)
    if args.attack_method == 'grb_injection':
        if args.attack_use_eot:
            return EOTGRBInjectionAttack(**attack_kwargs)
        return GRBInjectionAttack(**attack_kwargs)
    raise ValueError(f'unknown attack method: {args.attack_method}')


def save_clean_smoothing(output_dir, top2, count1, count2, clean_accuracy, smoothing_seconds, args):
    if args.save_artifacts:
        with open(os.path.join(output_dir, 'smoothing_result.pkl'), 'wb') as result_file:
            pickle.dump([top2, count1, count2], result_file)

    csv_path = os.path.join(output_dir, 'smoothing_result.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['metric', 'value'])
        csv_writer.writerow(['dataset', args.dataset])
        csv_writer.writerow(['model', args.model])
        csv_writer.writerow(['p_n', args.p_n])
        csv_writer.writerow(['n_smoothing', args.n_smoothing])
        csv_writer.writerow(['clean_accuracy', clean_accuracy])
        csv_writer.writerow(['smoothing_seconds', smoothing_seconds])
        csv_writer.writerow(['smoothing_minutes', smoothing_seconds / 60.0])
        csv_writer.writerow(['save_artifacts', args.save_artifacts])


def save_attack_results(output_dir, attack_eval, attack_meta, attack_info, attacked_features, attacked_edge_index, attack_seconds, args, restart_idx=None):
    graph_name = 'attacked_graph.pt' if restart_idx is None else f'attacked_graph_restart{restart_idx}.pt'
    if args.save_artifacts:
        torch.save(
            {
                'attack_method': args.attack_method,
                'attack_use_eot': args.attack_use_eot,
                'restart_idx': restart_idx,
                'features': attacked_features.detach().cpu(),
                'edge_index': attacked_edge_index.detach().cpu(),
                'injected_nodes': attack_meta.injected_nodes.detach().cpu(),
                'target_index': attack_meta.target_index.detach().cpu(),
                'attacked_top1': attack_eval.attacked_top1,
                'attacked_top2': attack_eval.attacked_top2,
                'attacked_count1': attack_eval.attacked_count1,
                'attacked_count2': attack_eval.attacked_count2,
            },
            os.path.join(output_dir, graph_name)
        )


def save_attack_restart_results(output_dir, restart_results):
    csv_path = os.path.join(output_dir, 'attack_result_restarts.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'restart_idx',
            'attack_seed',
            'eval_seed',
            'clean_accuracy',
            'attacked_accuracy',
            'attack_success_rate',
            'attack_seconds',
            'eval_count',
            'clean_correct_count',
            'injected_node_count',
            'added_edge_count',
            'attack_target_count',
            'ranking_rounds',
            'optimization_epochs',
            'sequential_rounds',
            'effective_attack_rank_samples',
            'effective_attack_opt_samples',
            'theoretical_model_calls',
        ])
        for result in restart_results:
            csv_writer.writerow([
                result['restart_idx'],
                result['attack_seed'],
                result['eval_seed'],
                result['clean_accuracy'],
                result['attacked_accuracy'],
                result['attack_success_rate'],
                result['attack_seconds'],
                result['eval_count'],
                result['clean_correct_count'],
                result['injected_node_count'],
                result['added_edge_count'],
                result['attack_target_count'],
                result['ranking_rounds'],
                result['optimization_epochs'],
                result['sequential_rounds'],
                result['effective_attack_rank_samples'],
                result['effective_attack_opt_samples'],
                result['theoretical_model_calls'],
            ])


def save_attack_summary(output_dir, restart_results, best_restart_result, args):
    attacked_accuracy_mean, attacked_accuracy_std = summarize_metric([result['attacked_accuracy'] for result in restart_results])
    attack_success_rate_mean, attack_success_rate_std = summarize_metric([result['attack_success_rate'] for result in restart_results])
    attack_seconds_mean, attack_seconds_std = summarize_metric([result['attack_seconds'] for result in restart_results])
    injected_node_count_mean, injected_node_count_std = summarize_metric([result['injected_node_count'] for result in restart_results])
    added_edge_count_mean, added_edge_count_std = summarize_metric([result['added_edge_count'] for result in restart_results])

    clean_accuracy = restart_results[0]['clean_accuracy'] if restart_results else 0.0
    eval_count = restart_results[0]['eval_count'] if restart_results else 0
    clean_correct_count = restart_results[0]['clean_correct_count'] if restart_results else 0

    csv_path = os.path.join(output_dir, 'attack_result.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['metric', 'value'])
        csv_writer.writerow(['attack_method', args.attack_method])
        csv_writer.writerow(['attack_use_eot', args.attack_use_eot])
        csv_writer.writerow(['attack_restarts', args.attack_restarts])
        csv_writer.writerow(['attack_seed', resolve_attack_seed(args)])
        csv_writer.writerow(['clean_accuracy', clean_accuracy])
        csv_writer.writerow(['attacked_accuracy_mean', attacked_accuracy_mean])
        csv_writer.writerow(['attacked_accuracy_std', attacked_accuracy_std])
        csv_writer.writerow(['attack_success_rate_mean', attack_success_rate_mean])
        csv_writer.writerow(['attack_success_rate_std', attack_success_rate_std])
        csv_writer.writerow(['attack_seconds_mean', attack_seconds_mean])
        csv_writer.writerow(['attack_seconds_std', attack_seconds_std])
        csv_writer.writerow(['eval_count', eval_count])
        csv_writer.writerow(['clean_correct_count', clean_correct_count])
        csv_writer.writerow(['injected_node_count_mean', injected_node_count_mean])
        csv_writer.writerow(['injected_node_count_std', injected_node_count_std])
        csv_writer.writerow(['added_edge_count_mean', added_edge_count_mean])
        csv_writer.writerow(['added_edge_count_std', added_edge_count_std])
        csv_writer.writerow(['attack_target_count', best_restart_result['attack_target_count']])
        csv_writer.writerow(['attack_n_inject_max', args.attack_n_inject_max])
        csv_writer.writerow(['attack_n_edge_max', args.attack_n_edge_max])
        csv_writer.writerow(['attack_lr', args.attack_lr])
        csv_writer.writerow(['attack_epochs', args.attack_epochs])
        csv_writer.writerow(['attack_sequential_step', args.attack_sequential_step])
        csv_writer.writerow(['attack_quiet', args.attack_quiet])
        csv_writer.writerow(['attack_clean_smoothing', args.attack_clean_smoothing])
        csv_writer.writerow(['attack_eval_smoothing', args.attack_eval_smoothing])
        csv_writer.writerow(['save_artifacts', args.save_artifacts])
        csv_writer.writerow(['configured_attack_rank_samples', args.attack_rank_samples])
        csv_writer.writerow(['configured_attack_opt_samples', args.attack_opt_samples])
        csv_writer.writerow(['effective_attack_rank_samples', best_restart_result['effective_attack_rank_samples']])
        csv_writer.writerow(['effective_attack_opt_samples', best_restart_result['effective_attack_opt_samples']])
        csv_writer.writerow(['attack_gnia_homophily', args.attack_gnia_homophily])
        csv_writer.writerow(['best_restart_idx', best_restart_result['restart_idx']])
        csv_writer.writerow(['best_restart_attack_success_rate', best_restart_result['attack_success_rate']])
        csv_writer.writerow(['best_restart_attacked_accuracy', best_restart_result['attacked_accuracy']])
        csv_writer.writerow(['best_restart_attack_seconds', best_restart_result['attack_seconds']])
        csv_writer.writerow(['best_restart_injected_node_count', best_restart_result['injected_node_count']])
        csv_writer.writerow(['best_restart_added_edge_count', best_restart_result['added_edge_count']])
        csv_writer.writerow(['best_restart_theoretical_model_calls', best_restart_result['theoretical_model_calls']])

    info_path = os.path.join(output_dir, 'attack_info.txt')
    with open(info_path, 'w') as info_file:
        info_file.write('defense_method: smooth\n')
        info_file.write(f'attack_method: {args.attack_method}\n')
        info_file.write(f'attack_use_eot: {args.attack_use_eot}\n')
        info_file.write(f'attack_restarts: {args.attack_restarts}\n')
        info_file.write(f'attack_seed: {resolve_attack_seed(args)}\n')
        info_file.write(f'clean_accuracy: {clean_accuracy:.6f}\n')
        info_file.write(f'attacked_accuracy_mean: {attacked_accuracy_mean:.6f}\n')
        info_file.write(f'attacked_accuracy_std: {attacked_accuracy_std:.6f}\n')
        info_file.write(f'attack_success_rate_mean: {attack_success_rate_mean:.6f}\n')
        info_file.write(f'attack_success_rate_std: {attack_success_rate_std:.6f}\n')
        info_file.write(f'attack_seconds_mean: {attack_seconds_mean:.6f}\n')
        info_file.write(f'attack_seconds_std: {attack_seconds_std:.6f}\n')
        info_file.write(f'injected_node_count_mean: {injected_node_count_mean:.6f}\n')
        info_file.write(f'injected_node_count_std: {injected_node_count_std:.6f}\n')
        info_file.write(f'added_edge_count_mean: {added_edge_count_mean:.6f}\n')
        info_file.write(f'added_edge_count_std: {added_edge_count_std:.6f}\n')
        info_file.write(f'attack_target_count: {best_restart_result["attack_target_count"]}\n')
        info_file.write(f'attack_n_inject_max: {args.attack_n_inject_max}\n')
        info_file.write(f'attack_n_edge_max: {args.attack_n_edge_max}\n')
        info_file.write(f'attack_lr: {args.attack_lr}\n')
        info_file.write(f'attack_epochs: {args.attack_epochs}\n')
        info_file.write(f'attack_sequential_step: {args.attack_sequential_step}\n')
        info_file.write(f'attack_quiet: {args.attack_quiet}\n')
        info_file.write(f'attack_clean_smoothing: {args.attack_clean_smoothing}\n')
        info_file.write(f'attack_eval_smoothing: {args.attack_eval_smoothing}\n')
        info_file.write(f'save_artifacts: {args.save_artifacts}\n')
        info_file.write(f'attack_gnia_homophily: {args.attack_gnia_homophily}\n')
        info_file.write(f'best_restart_idx: {best_restart_result["restart_idx"]}\n')
        info_file.write(f'best_restart_attack_success_rate: {best_restart_result["attack_success_rate"]:.6f}\n')
        info_file.write(f'best_restart_attacked_accuracy: {best_restart_result["attacked_accuracy"]:.6f}\n')
        info_file.write(f'best_restart_injected_node_count: {best_restart_result["injected_node_count"]}\n')
        info_file.write(f'best_restart_added_edge_count: {best_restart_result["added_edge_count"]}\n')
        info_file.write(f'configured_attack_rank_samples: {args.attack_rank_samples}\n')
        info_file.write(f'configured_attack_opt_samples: {args.attack_opt_samples}\n')
        info_file.write(f'effective_attack_rank_samples: {best_restart_result["effective_attack_rank_samples"]}\n')
        info_file.write(f'effective_attack_opt_samples: {best_restart_result["effective_attack_opt_samples"]}\n')
        info_file.write(f'best_restart_theoretical_model_calls: {best_restart_result["theoretical_model_calls"]}\n')


def better_restart(lhs, rhs):
    if rhs is None:
        return True
    lhs_key = (lhs['attack_success_rate'], -lhs['attacked_accuracy'], -lhs['attack_seconds'])
    rhs_key = (rhs['attack_success_rate'], -rhs['attacked_accuracy'], -rhs['attack_seconds'])
    return lhs_key > rhs_key

    csv_path = os.path.join(output_dir, 'attack_result.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['metric', 'value'])
        csv_writer.writerow(['attack_method', args.attack_method])
        csv_writer.writerow(['attack_use_eot', args.attack_use_eot])
        csv_writer.writerow(['clean_accuracy', attack_eval.clean_accuracy])
        csv_writer.writerow(['attacked_accuracy', attack_eval.attacked_accuracy])
        csv_writer.writerow(['attack_success_rate', attack_eval.asr])
        csv_writer.writerow(['attack_seconds', attack_seconds])
        csv_writer.writerow(['attack_minutes', attack_seconds / 60.0])
        csv_writer.writerow(['eval_count', attack_eval.eval_count])
        csv_writer.writerow(['clean_correct_count', attack_eval.clean_correct_count])
        csv_writer.writerow(['injected_node_count', len(attack_meta.injected_nodes)])
        csv_writer.writerow(['attack_target_count', attack_meta.target_count])
        csv_writer.writerow(['attack_clean_smoothing', args.attack_clean_smoothing])
        csv_writer.writerow(['attack_eval_smoothing', args.attack_eval_smoothing])
        csv_writer.writerow(['configured_attack_rank_samples', args.attack_rank_samples])
        csv_writer.writerow(['configured_attack_opt_samples', args.attack_opt_samples])
        csv_writer.writerow(['effective_attack_rank_samples', attack_info.n_rank_samples])
        csv_writer.writerow(['effective_attack_opt_samples', attack_info.n_opt_samples])
        csv_writer.writerow(['attack_gnia_homophily', args.attack_gnia_homophily])
        csv_writer.writerow(['ranking_rounds', attack_info.ranking_rounds])
        csv_writer.writerow(['optimization_epochs', attack_info.optimization_epochs])
        csv_writer.writerow(['sequential_rounds', attack_info.sequential_rounds])
        csv_writer.writerow(['theoretical_model_calls', attack_info.theoretical_model_calls])

    info_path = os.path.join(output_dir, 'attack_info.txt')
    with open(info_path, 'w') as info_file:
        info_file.write(f'attack_method: {args.attack_method}\n')
        info_file.write(f'attack_use_eot: {args.attack_use_eot}\n')
        info_file.write(f'clean_accuracy: {attack_eval.clean_accuracy:.6f}\n')
        info_file.write(f'attacked_accuracy: {attack_eval.attacked_accuracy:.6f}\n')
        info_file.write(f'attack_success_rate: {attack_eval.asr:.6f}\n')
        info_file.write(f'attack_seconds: {attack_seconds:.6f}\n')
        info_file.write(f'injected_node_count: {len(attack_meta.injected_nodes)}\n')
        info_file.write(f'attack_target_count: {attack_meta.target_count}\n')
        info_file.write(f'configured_attack_rank_samples: {args.attack_rank_samples}\n')
        info_file.write(f'configured_attack_opt_samples: {args.attack_opt_samples}\n')
        info_file.write(f'effective_attack_rank_samples: {attack_info.n_rank_samples}\n')
        info_file.write(f'effective_attack_opt_samples: {attack_info.n_opt_samples}\n')


def prepare_output_paths(args):
    sample_config = {'p_n': args.p_n}
    attack_tag = format_attack_variant(args.attack_method, args.attack_use_eot)
    default_model_dir = os.path.join(
        f'{args.model_dir}/{args.dataset}_{args.model}_{sample_config["p_n"]}_seed{args.seed}',
        'model.pth',
    )
    args.model_dir = default_model_dir
    model_parent_dir = os.path.dirname(args.model_dir) or '.'
    args.output_dir = (
        f'{args.output_dir}/smooth_{attack_tag}_{args.dataset}_{args.model}_pn{sample_config["p_n"]}'
        f'_seed{args.seed}_eval{args.attack_eval_smoothing}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(model_parent_dir, exist_ok=True)
    return sample_config


def main(args):
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpuID}')
        print(f'---using GPU---cuda:{args.gpuID}----')
    else:
        print('---using CPU---')
        args.device = torch.device('cpu')

    init_random_seed(args.seed)
    sample_config = prepare_output_paths(args)

    features, edge_index, labels, train_mask, val_mask, test_mask, num_x, num_labels = load_node_dataset(args)
    model = build_smoothed_model(args, num_x, num_labels, sample_config)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if not os.path.exists(args.model_dir) or args.force_training:
        train_start = time.time()
        train_smoothing_model_nc(model, features, edge_index, labels, train_mask, val_mask, test_mask, optimizer, args)
        train_seconds = time.time() - train_start
        with open(os.path.join(args.output_dir, 'training_time.txt'), 'w') as timing_file:
            timing_file.write(f'training_seconds: {train_seconds:.6f}\n')
            timing_file.write(f'training_minutes: {train_seconds / 60.0:.6f}\n')

    if os.path.exists(args.model_dir):
        model = torch.load(args.model_dir, map_location=args.device, weights_only=False)
        model.to(args.device)

    model.eval()
    node_no_dense = args.dataset == 'Amazon'

    clean_start = time.time()
    top2, count1, count2 = model.smoothed_precit(
        features,
        edge_index,
        num=args.n_smoothing,
        no_dense=node_no_dense,
    )
    clean_seconds = time.time() - clean_start

    test_index = torch.where(test_mask)[0].cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    clean_correct = np.asarray(top2[test_index, 0]) == labels_np[test_index]
    clean_accuracy = float(clean_correct.mean()) if test_index.size > 0 else 0.0
    print(f'Smoothed classifier accuracy: {clean_accuracy:.6f}')
    save_clean_smoothing(args.output_dir, top2, count1, count2, clean_accuracy, clean_seconds, args)

    attack_clean_smoothing = args.attack_clean_smoothing
    if attack_clean_smoothing != args.n_smoothing:
        print(f'Recompute clean smoothing for attack target selection with {attack_clean_smoothing} samples')
        attack_clean_top2, _, _ = model.smoothed_precit(
            features,
            edge_index,
            num=attack_clean_smoothing,
            no_dense=node_no_dense,
        )
    else:
        attack_clean_top2 = top2

    clean_top1 = torch.as_tensor(attack_clean_top2[:, 0], device=args.device, dtype=torch.long)
    attack_target_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=args.device)
    attack_target_mask[test_mask] = clean_top1[test_mask] == labels[test_mask]
    print(f'Attack mode: {"EOT" if args.attack_use_eot else "non-EOT"}')

    attack_seed = resolve_attack_seed(args)
    restart_results = []
    best_restart_result = None

    for restart_idx in range(args.attack_restarts):
        current_attack_seed = restart_seed(attack_seed, restart_idx)
        current_eval_seed = restart_seed(attack_seed, restart_idx, offset=1000000)
        print(
            f'Attack restart {restart_idx + 1}/{args.attack_restarts} '
            f'(attack_seed={current_attack_seed}, eval_seed={current_eval_seed})'
        )

        set_attack_random_seed(current_attack_seed)
        attack = build_attack(args, features)
        attack_start = time.time()
        attacked_features, attacked_edge_index, attack_meta, attack_info = attack.attack(
            model=model,
            features=features,
            edge_index=edge_index,
            anchor_labels=clean_top1,
            target_mask=attack_target_mask,
            no_dense=node_no_dense,
        )
        attack_seconds = time.time() - attack_start

        set_attack_random_seed(current_eval_seed)
        attack_eval = evaluate_smoothed_attack(
            model=model,
            clean_features=features,
            clean_edge_index=edge_index,
            attacked_features=attacked_features,
            attacked_edge_index=attacked_edge_index,
            labels=labels,
            test_mask=test_mask,
            num_smoothing=args.attack_eval_smoothing,
            no_dense=node_no_dense,
            clean_top1=attack_clean_top2[:, 0],
        )

        restart_result = {
            'restart_idx': restart_idx,
            'attack_seed': current_attack_seed,
            'eval_seed': current_eval_seed,
            'clean_accuracy': attack_eval.clean_accuracy,
            'attacked_accuracy': attack_eval.attacked_accuracy,
            'attack_success_rate': attack_eval.asr,
            'attack_seconds': attack_seconds,
            'eval_count': attack_eval.eval_count,
            'clean_correct_count': attack_eval.clean_correct_count,
            'injected_node_count': len(attack_meta.injected_nodes),
            'added_edge_count': int(attacked_edge_index.shape[1] - edge_index.shape[1]),
            'attack_target_count': attack_meta.target_count,
            'ranking_rounds': attack_info.ranking_rounds,
            'optimization_epochs': attack_info.optimization_epochs,
            'sequential_rounds': attack_info.sequential_rounds,
            'effective_attack_rank_samples': attack_info.n_rank_samples,
            'effective_attack_opt_samples': attack_info.n_opt_samples,
            'theoretical_model_calls': attack_info.theoretical_model_calls,
        }
        restart_results.append(restart_result)

        print(f'{args.attack_method} attack restart {restart_idx + 1} finished in {attack_seconds:.2f} seconds')
        print(f'Attacked smoothed accuracy: {attack_eval.attacked_accuracy:.6f}')
        print(f'Attack success rate: {attack_eval.asr:.6f}')

        if better_restart(restart_result, best_restart_result):
            best_restart_result = restart_result
            save_attack_results(
                args.output_dir,
                attack_eval,
                attack_meta,
                attack_info,
                attacked_features,
                attacked_edge_index,
                attack_seconds,
                args,
                restart_idx=restart_idx,
            )
            save_attack_results(
                args.output_dir,
                attack_eval,
                attack_meta,
                attack_info,
                attacked_features,
                attacked_edge_index,
                attack_seconds,
                args,
            )

    save_attack_restart_results(args.output_dir, restart_results)
    save_attack_summary(args.output_dir, restart_results, best_restart_result, args)
    print(f'Results saved to {args.output_dir}')
    return {
        'best_restart': best_restart_result,
        'attack_success_rate_mean': summarize_metric([result['attack_success_rate'] for result in restart_results])[0],
        'attack_success_rate_std': summarize_metric([result['attack_success_rate'] for result in restart_results])[1],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate smoothed node classifiers with node injection attacks')

    parser.add_argument('-gpuID', type=int, default=0)
    parser.add_argument('-seed', type=int, default=2020)
    parser.add_argument('-attack_seed', type=int, default=None, help='base random seed for attack restarts; defaults to seed')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-clip_max', type=float, default=2.0, help='gradient clipping max norm')
    parser.add_argument('-criterion', type=str, default='nll', choices=['nll', 'ce'], help='loss function')
    parser.add_argument('-patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('-epochs', type=int, default=1000, help='training epoch')
    parser.add_argument('-save_model', action='store_true', default=True, help='save model')
    parser.add_argument('-dataset', type=str, default='CiteSeer', choices=['PubMed', 'CiteSeer', 'computers', 'Cora-ML', 'Amazon'])
    parser.add_argument('-model', type=str, default='GAT', choices=['GCN', 'GAT', 'GSAGE', 'GIN', 'APPNP'], help='GNN model')
    parser.add_argument('-n_hidden', type=int, default=64, help='size of hidden layer')
    parser.add_argument('-drop', type=float, default=0.1, help='dropout rate')
    parser.add_argument('-weight_decay', type=float, default=0, help='weight decay rate')
    parser.add_argument('-force_training', action='store_true', default=False, help='force training even if pretrained model exists')

    parser.add_argument('-p_n', type=float, default=0.9, help='probability of deleting nodes')
    parser.add_argument('-n_smoothing', type=int, default=10000, help='number of smoothing samples for clean evaluation')

    parser.add_argument('-output_dir', type=str, default='./results_adv_test', help='output directory')
    parser.add_argument(
        '-model_dir',
        type=str,
        default='./models',
        help='model root directory; checkpoint path resolves to <model_dir>/<dataset>_<model>_<p_n>_seed<seed>/model.pth',
    )
    parser.add_argument('-save_artifacts', action='store_true', default=False, help='save large artifacts (smoothing_result.pkl and attacked_graph*.pt); disabled by default')

    parser.add_argument('-attack_method', type=str, default='tdgia', choices=['tdgia', 'gnia', 'grb_injection'], help='node injection attack to run')
    parser.add_argument('-attack_use_eot', type=parse_bool_arg, default=True, help='whether to use EOT attack optimization against the smoothed model')
    parser.add_argument('-attack_restarts', type=int, default=1, help='number of random attack restarts for estimating mean/std attack performance')
    parser.add_argument('-attack_n_inject_max', type=int, default=20, help='maximum number of injected nodes')
    parser.add_argument('-attack_n_edge_max', type=int, default=5, help='maximum number of edges for each injected node')
    parser.add_argument('-attack_lr', type=float, default=0.01, help='learning rate for injected feature optimization')
    parser.add_argument('-attack_epochs', type=int, default=100, help='optimization epochs for injected node features')
    parser.add_argument('-attack_rank_samples', type=int, default=16, help='number of EOT samples used for topology ranking')
    parser.add_argument('-attack_opt_samples', type=int, default=8, help='number of EOT samples used in each optimization step')
    parser.add_argument('-attack_gnia_homophily', type=float, default=0.5, help='homophily regularization weight for the G-NIA style attack')
    parser.add_argument('-attack_clean_smoothing', type=int, default=10000, help='number of smoothing samples used for clean target selection in attack mode')
    parser.add_argument('-attack_eval_smoothing', type=int, default=10000, help='number of smoothing samples used to evaluate the attacked graph')
    parser.add_argument('-attack_sequential_step', type=float, default=0.2, help='fraction of injected nodes added per sequential attack step')
    parser.add_argument('-attack_quiet', action='store_true', default=False, help='disable per-step attack progress logs')

    main(parser.parse_args())