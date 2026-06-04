import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from attacks.gnia import GNIAAttack
from attacks.grb_injection import GRBInjectionAttack
from attacks.injection_common import evaluate_standard_attack
from attacks.tdgia import TDGIAAttack
from dataset_loader import load_node_data
from models import NodeAPPNP, NodeGAT, NodeGCN, NodeGIN, NodeGSAGE
from utils import accuracy


def init_random_seed(seed=2021):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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

        datasets = PygNodePropPredDataset(name='ogbn-products')
        graph = datasets[0]
        features = graph.x.to(args.device)
        edge_index = graph.edge_index.to(args.device)
        labels = graph.y.squeeze().to(args.device)
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


def build_node_model(args, num_x, num_labels):
    if args.model == 'GCN':
        return NodeGCN(num_x, num_labels, hidden_size=args.n_hidden).to(args.device)
    if args.model == 'GAT':
        return NodeGAT(num_x, num_labels, hidden_size=args.n_hidden).to(args.device)
    if args.model == 'GSAGE':
        return NodeGSAGE(num_x, num_labels, hidden_size=args.n_hidden).to(args.device)
    if args.model == 'GIN':
        return NodeGIN(num_x, num_labels, hidden_size=args.n_hidden).to(args.device)
    if args.model == 'APPNP':
        return NodeAPPNP(num_x, num_labels, hidden_size=args.n_hidden).to(args.device)
    raise ValueError(f'unknown model: {args.model}')


def resolve_effective_attack_samples(args):
    effective_rank_samples = 1
    effective_opt_samples = 1
    configured_pair = (args.attack_rank_samples, args.attack_opt_samples)
    effective_pair = (effective_rank_samples, effective_opt_samples)
    if configured_pair != effective_pair and not getattr(args, '_warned_standard_attack_samples', False):
        print(
            'attack_rank_samples and attack_opt_samples are compatibility-only '
            'for standard attacks; using one forward pass for ranking and optimization.'
        )
        args._warned_standard_attack_samples = True
    return effective_rank_samples, effective_opt_samples


def build_attack(method, args, features):
    effective_rank_samples, effective_opt_samples = resolve_effective_attack_samples(args)
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
    if method == 'tdgia':
        return TDGIAAttack(**attack_kwargs)
    if method == 'gnia':
        return GNIAAttack(**attack_kwargs, homophily_weight=args.attack_gnia_homophily)
    if method == 'grb_injection':
        return GRBInjectionAttack(**attack_kwargs)
    raise ValueError(f'unknown attack method: {method}')


def select_attack_targets(model, features, edge_index, labels, candidate_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        clean_top1 = logits.argmax(dim=1)
    target_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    target_mask[candidate_mask] = clean_top1[candidate_mask] == labels[candidate_mask]
    return clean_top1, target_mask


def evaluate_clean_split(model, features, edge_index, labels, split_mask, criterion_name):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        if criterion_name == 'nll':
            loss = F.nll_loss(F.log_softmax(logits[split_mask], dim=1), labels[split_mask])
        elif criterion_name == 'ce':
            loss = F.cross_entropy(logits[split_mask], labels[split_mask])
        else:
            raise ValueError(f'unknown criterion: {criterion_name}')
        acc = accuracy(logits[split_mask], labels[split_mask])
    return float(loss.item()), float(acc.item())


def save_checkpoint(model, args, num_x, num_labels):
    torch.save(
        {
            'state_dict': model.state_dict(),
            'model_name': args.model,
            'num_x': num_x,
            'num_labels': num_labels,
            'n_hidden': args.n_hidden,
        },
        args.model_dir,
    )


def load_checkpoint(args, num_x, num_labels):
    checkpoint = torch.load(args.model_dir, map_location=args.device, weights_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model = build_node_model(args, num_x, num_labels)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)
        return model

    # Backward compatibility for legacy checkpoints saved via torch.save(model, ...).
    model = torch.load(args.model_dir, map_location=args.device, weights_only=False)
    model.to(args.device)
    return model


def format_attack_variant(method, use_eot=False):
    prefix = 'eot' if use_eot else 'base'
    return f'{prefix}_{method}'


def save_attack_results(output_dir, attack_eval, attack_meta, attack_info, attacked_features, attacked_edge_index, clean_edge_index, attack_seconds, args):
    added_edge_count = int(attacked_edge_index.shape[1] - clean_edge_index.shape[1])
    if args.save_artifacts:
        torch.save(
            {
                'train_attack_method': args.train_attack_method,
                'eval_attack_method': args.eval_attack_method,
                'features': attacked_features.detach().cpu(),
                'edge_index': attacked_edge_index.detach().cpu(),
                'injected_nodes': attack_meta.injected_nodes.detach().cpu(),
                'target_index': attack_meta.target_index.detach().cpu(),
                'attacked_top1': attack_eval.attacked_top1,
                'attacked_top2': attack_eval.attacked_top2,
                'attacked_score1': attack_eval.attacked_count1,
                'attacked_score2': attack_eval.attacked_count2,
            },
            os.path.join(output_dir, 'attacked_graph.pt')
        )

    csv_path = os.path.join(output_dir, 'attack_result.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['metric', 'value'])
        csv_writer.writerow(['train_attack_method', args.train_attack_method])
        csv_writer.writerow(['eval_attack_method', args.eval_attack_method])
        csv_writer.writerow(['clean_accuracy', attack_eval.clean_accuracy])
        csv_writer.writerow(['attacked_accuracy', attack_eval.attacked_accuracy])
        csv_writer.writerow(['attack_success_rate', attack_eval.asr])
        csv_writer.writerow(['attack_seconds', attack_seconds])
        csv_writer.writerow(['attack_minutes', attack_seconds / 60.0])
        csv_writer.writerow(['eval_count', attack_eval.eval_count])
        csv_writer.writerow(['clean_correct_count', attack_eval.clean_correct_count])
        csv_writer.writerow(['injected_node_count', len(attack_meta.injected_nodes)])
        csv_writer.writerow(['added_edge_count', added_edge_count])
        csv_writer.writerow(['attack_target_count', attack_meta.target_count])
        csv_writer.writerow(['attack_n_inject_max', args.attack_n_inject_max])
        csv_writer.writerow(['attack_n_edge_max', args.attack_n_edge_max])
        csv_writer.writerow(['attack_lr', args.attack_lr])
        csv_writer.writerow(['attack_epochs', args.attack_epochs])
        csv_writer.writerow(['attack_gnia_homophily', args.attack_gnia_homophily])
        csv_writer.writerow(['attack_sequential_step', args.attack_sequential_step])
        csv_writer.writerow(['attack_quiet', args.attack_quiet])
        csv_writer.writerow(['save_artifacts', args.save_artifacts])
        csv_writer.writerow(['configured_attack_rank_samples', args.attack_rank_samples])
        csv_writer.writerow(['configured_attack_opt_samples', args.attack_opt_samples])
        csv_writer.writerow(['effective_attack_rank_samples', attack_info.n_rank_samples])
        csv_writer.writerow(['effective_attack_opt_samples', attack_info.n_opt_samples])
        csv_writer.writerow(['ranking_rounds', attack_info.ranking_rounds])
        csv_writer.writerow(['optimization_epochs', attack_info.optimization_epochs])
        csv_writer.writerow(['sequential_rounds', attack_info.sequential_rounds])
        csv_writer.writerow(['theoretical_model_calls', attack_info.theoretical_model_calls])

    info_path = os.path.join(output_dir, 'attack_info.txt')
    with open(info_path, 'w') as info_file:
        info_file.write(f'defense_method: adv_{args.train_attack_method}\n')
        info_file.write(f'train_attack_method: {args.train_attack_method}\n')
        info_file.write(f'attack_method: {args.eval_attack_method}\n')
        info_file.write(f'attack_use_eot: False\n')
        info_file.write(f'attack_restarts: 1\n')
        info_file.write(f'attack_seed: {args.seed}\n')
        info_file.write(f'clean_accuracy: {attack_eval.clean_accuracy:.6f}\n')
        info_file.write(f'attacked_accuracy_mean: {attack_eval.attacked_accuracy:.6f}\n')
        info_file.write(f'attacked_accuracy_std: 0.000000\n')
        info_file.write(f'attack_success_rate_mean: {attack_eval.asr:.6f}\n')
        info_file.write(f'attack_success_rate_std: 0.000000\n')
        info_file.write(f'attack_seconds_mean: {attack_seconds:.6f}\n')
        info_file.write(f'attack_seconds_std: 0.000000\n')
        info_file.write(f'injected_node_count_mean: {float(len(attack_meta.injected_nodes)):.6f}\n')
        info_file.write(f'injected_node_count_std: 0.000000\n')
        info_file.write(f'added_edge_count_mean: {float(added_edge_count):.6f}\n')
        info_file.write(f'added_edge_count_std: 0.000000\n')
        info_file.write(f'attack_target_count: {attack_meta.target_count}\n')
        info_file.write(f'attack_n_inject_max: {args.attack_n_inject_max}\n')
        info_file.write(f'attack_n_edge_max: {args.attack_n_edge_max}\n')
        info_file.write(f'attack_lr: {args.attack_lr}\n')
        info_file.write(f'attack_epochs: {args.attack_epochs}\n')
        info_file.write(f'attack_gnia_homophily: {args.attack_gnia_homophily}\n')
        info_file.write(f'attack_sequential_step: {args.attack_sequential_step}\n')
        info_file.write(f'attack_quiet: {args.attack_quiet}\n')
        info_file.write(f'attack_clean_smoothing: NA\n')
        info_file.write(f'attack_eval_smoothing: NA\n')
        info_file.write(f'save_artifacts: {args.save_artifacts}\n')
        info_file.write(f'best_restart_idx: 0\n')
        info_file.write(f'best_restart_attack_success_rate: {attack_eval.asr:.6f}\n')
        info_file.write(f'best_restart_attacked_accuracy: {attack_eval.attacked_accuracy:.6f}\n')
        info_file.write(f'best_restart_attack_seconds: {attack_seconds:.6f}\n')
        info_file.write(f'best_restart_injected_node_count: {len(attack_meta.injected_nodes)}\n')
        info_file.write(f'best_restart_added_edge_count: {added_edge_count}\n')
        info_file.write(f'configured_attack_rank_samples: {args.attack_rank_samples}\n')
        info_file.write(f'configured_attack_opt_samples: {args.attack_opt_samples}\n')
        info_file.write(f'effective_attack_rank_samples: {attack_info.n_rank_samples}\n')
        info_file.write(f'effective_attack_opt_samples: {attack_info.n_opt_samples}\n')
        info_file.write(f'ranking_rounds: {attack_info.ranking_rounds}\n')
        info_file.write(f'optimization_epochs: {attack_info.optimization_epochs}\n')
        info_file.write(f'sequential_rounds: {attack_info.sequential_rounds}\n')
        info_file.write(f'theoretical_model_calls: {attack_info.theoretical_model_calls}\n')
        info_file.write(f'best_restart_theoretical_model_calls: {attack_info.theoretical_model_calls}\n')


def prepare_output_paths(args):
    defense_tag = f'adv_{args.train_attack_method}'
    attack_tag = format_attack_variant(args.eval_attack_method)
    default_model_dir = os.path.join(
        args.model_dir,
        f'{args.dataset}_{args.model}_adv_{args.train_attack_method}_seed{args.seed}',
        'model.pth',
    )
    args.model_dir = default_model_dir
    model_parent_dir = os.path.dirname(args.model_dir) or '.'
    args.output_dir = os.path.join(
        args.output_dir,
        f'{defense_tag}_{attack_tag}_{args.dataset}_{args.model}'
        f'_seed{args.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(model_parent_dir, exist_ok=True)


def adversarial_train_node_model(model, features, edge_index, labels, train_mask, val_mask, optimizer, attack, args, num_x, num_labels):
    endure_count = 0
    best_acc_val = 0.0
    train_csv_path = os.path.join(args.output_dir, 'train_metrics.csv')
    train_result_path = os.path.join(args.output_dir, 'train_result.txt')
    train_index = torch.where(train_mask)[0]

    with open(train_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'loss_clean', 'loss_adv', 'loss_total', 'acc_train',
            'loss_val', 'acc_val', 'adv_target_count', 'seconds'
        ])

    for epoch in range(args.epochs):
        t_start = time.time()
        model.train()
        optimizer.zero_grad()

        logits_clean = model(features, edge_index)
        if args.criterion == 'nll':
            loss_clean = F.nll_loss(F.log_softmax(logits_clean[train_mask], dim=1), labels[train_mask])
        elif args.criterion == 'ce':
            loss_clean = F.cross_entropy(logits_clean[train_mask], labels[train_mask])
        else:
            raise ValueError(f'unknown criterion: {args.criterion}')
        acc_train = accuracy(logits_clean[train_mask], labels[train_mask])

        loss_adv = None
        adv_target_count = 0
        loss_total = loss_clean

        should_attack = epoch >= args.adv_warmup and ((epoch - args.adv_warmup) % args.adv_attack_interval == 0)
        if should_attack:
            clean_top1, attack_target_mask = select_attack_targets(model, features, edge_index, labels, train_mask)
            adv_target_count = int(attack_target_mask.sum().item())
            if adv_target_count > 0:
                attacked_features, attacked_edge_index, _, _ = attack.attack(
                    model=model,
                    features=features,
                    edge_index=edge_index,
                    anchor_labels=clean_top1,
                    target_mask=attack_target_mask,
                )
                model.zero_grad(set_to_none=True)
                model.train()
                logits_adv = model(attacked_features, attacked_edge_index)
                if args.criterion == 'nll':
                    loss_adv = F.nll_loss(F.log_softmax(logits_adv[train_index], dim=1), labels[train_index])
                else:
                    loss_adv = F.cross_entropy(logits_adv[train_index], labels[train_index])
                loss_total = loss_clean + args.adv_weight * loss_adv

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max)
        optimizer.step()

        if epoch % args.eval_interval == 0:
            loss_val, acc_val = evaluate_clean_split(model, features, edge_index, labels, val_mask, args.criterion)
            loss_adv_value = '' if loss_adv is None else f'{loss_adv.item():.6f}'
            print(
                'Epoch: {:04d}'.format(epoch + 1),
                'loss_clean: {:.4f}'.format(loss_clean.item()),
                'loss_adv: {}'.format(loss_adv_value if loss_adv_value else 'NA'),
                'loss_total: {:.4f}'.format(loss_total.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val),
                'acc_val: {:.4f}'.format(acc_val),
                'adv_targets: {:d}'.format(adv_target_count),
                'time: {:.4f}s'.format(time.time() - t_start),
            )
            with open(train_csv_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    epoch + 1,
                    float(loss_clean.item()),
                    '' if loss_adv is None else float(loss_adv.item()),
                    float(loss_total.item()),
                    float(acc_train.item()),
                    loss_val,
                    acc_val,
                    adv_target_count,
                    time.time() - t_start,
                ])
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                endure_count = 0
                if args.save_model:
                    save_checkpoint(model, args, num_x, num_labels)
                    print('model saved to:', args.model_dir)
            else:
                endure_count += 1
            if endure_count > args.patience:
                print('early stop at epoch:', epoch)
                break

    with open(train_result_path, 'w') as txt_file:
        txt_file.write(f'train_attack_method: {args.train_attack_method}\n')
        txt_file.write(f'adv_weight: {args.adv_weight}\n')
        txt_file.write(f'adv_warmup: {args.adv_warmup}\n')
        txt_file.write(f'best_acc_val: {best_acc_val:.6f}\n')


def main(args):
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpuID}')
        print(f'---using GPU---cuda:{args.gpuID}----')
    else:
        args.device = torch.device('cpu')
        print('---using CPU---')

    init_random_seed(args.seed)
    prepare_output_paths(args)
    features, edge_index, labels, train_mask, val_mask, test_mask, num_x, num_labels = load_node_dataset(args)

    model = build_node_model(args, num_x, num_labels)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_attack = build_attack(args.train_attack_method, args, features)

    if not os.path.exists(args.model_dir) or args.force_training:
        train_start = time.time()
        adversarial_train_node_model(
            model,
            features,
            edge_index,
            labels,
            train_mask,
            val_mask,
            optimizer,
            train_attack,
            args,
            num_x,
            num_labels,
        )
        train_seconds = time.time() - train_start
        with open(os.path.join(args.output_dir, 'training_time.txt'), 'w') as train_time_file:
            train_time_file.write(f'training_seconds: {train_seconds:.6f}\n')
            train_time_file.write(f'training_minutes: {train_seconds / 60.0:.6f}\n')

    if os.path.exists(args.model_dir):
        model = load_checkpoint(args, num_x, num_labels)

    clean_loss_test, clean_acc_test = evaluate_clean_split(model, features, edge_index, labels, test_mask, args.criterion)
    print(f'Clean test loss: {clean_loss_test:.6f}')
    print(f'Clean test accuracy: {clean_acc_test:.6f}')

    clean_top1, attack_target_mask = select_attack_targets(model, features, edge_index, labels, test_mask)
    eval_attack = build_attack(args.eval_attack_method, args, features)

    attack_start = time.time()
    attacked_features, attacked_edge_index, attack_meta, attack_info = eval_attack.attack(
        model=model,
        features=features,
        edge_index=edge_index,
        anchor_labels=clean_top1,
        target_mask=attack_target_mask,
    )
    attack_seconds = time.time() - attack_start

    attack_eval = evaluate_standard_attack(
        model=model,
        clean_features=features,
        clean_edge_index=edge_index,
        attacked_features=attacked_features,
        attacked_edge_index=attacked_edge_index,
        labels=labels,
        test_mask=test_mask,
        clean_top1=clean_top1.detach().cpu().numpy(),
    )
    print(f'Attacked test accuracy: {attack_eval.attacked_accuracy:.6f}')
    print(f'Attack success rate: {attack_eval.asr:.6f}')

    save_attack_results(
        args.output_dir,
        attack_eval,
        attack_meta,
        attack_info,
        attacked_features,
        attacked_edge_index,
        edge_index,
        attack_seconds,
        args,
    )
    return attack_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='adversarial training and evaluation for node classification GNNs')

    parser.add_argument('-gpuID', type=int, default=0)
    parser.add_argument('-seed', type=int, default=2020)
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-clip_max', type=float, default=2.0, help='gradient clipping max norm')
    parser.add_argument('-criterion', type=str, default='ce', choices=['nll', 'ce'], help='loss function')
    parser.add_argument('-patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('-epochs', type=int, default=300, help='training epochs')
    parser.set_defaults(save_model=True)
    parser.add_argument('-save_model', dest='save_model', action='store_true', help='save model checkpoint')
    parser.add_argument('-no_save_model', dest='save_model', action='store_false', help='disable model checkpoint saving')
    parser.add_argument('-dataset', type=str, default='CiteSeer', choices=['PubMed', 'CiteSeer', 'computers', 'Cora-ML', 'Amazon'])
    parser.add_argument('-model', type=str, default='GCN', choices=['GCN', 'GAT', 'GSAGE', 'GIN', 'APPNP'], help='GNN model')
    parser.add_argument('-n_hidden', type=int, default=64, help='hidden size')
    parser.add_argument('-weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('-force_training', action='store_true', default=False, help='force retraining even if model exists')
    parser.add_argument('-eval_interval', type=int, default=10, help='validation interval during training')

    parser.add_argument('-adv_weight', type=float, default=1.0, help='weight for adversarial training loss')
    parser.add_argument('-adv_warmup', type=int, default=0, help='number of clean-only warmup epochs')
    parser.add_argument('-adv_attack_interval', type=int, default=1, help='frequency of adversarial attack generation during training')
    parser.add_argument('-train_attack_method', type=str, default='tdgia', choices=['tdgia', 'gnia', 'grb_injection'], help='attack used to build adversarial training samples')
    parser.add_argument('-eval_attack_method', type=str, default='tdgia', choices=['tdgia', 'gnia', 'grb_injection'], help='attack used for test-time robustness evaluation')

    parser.add_argument('-output_dir', type=str, default='./results_adv', help='output directory')
    parser.add_argument(
        '-model_dir',
        '-model_root',
        dest='model_dir',
        type=str,
        default='./models',
        help='model root directory; checkpoint path resolves to <model_dir>/<dataset>_<model>_adv_<train_attack_method>_seed<seed>/model.pth',
    )
    parser.add_argument('-save_artifacts', action='store_true', default=False, help='save large attack artifacts (attacked_graph.pt); disabled by default')

    parser.add_argument('-attack_n_inject_max', type=int, default=20, help='maximum number of injected nodes')
    parser.add_argument('-attack_n_edge_max', type=int, default=5, help='maximum number of edges per injected node')
    parser.add_argument('-attack_lr', type=float, default=0.01, help='learning rate for injected features')
    parser.add_argument('-attack_epochs', type=int, default=100, help='optimization epochs for injected features')
    parser.add_argument('-attack_rank_samples', type=int, default=1, help='kept for argument compatibility; regular attacks use one forward pass')
    parser.add_argument('-attack_opt_samples', type=int, default=1, help='kept for argument compatibility; regular attacks use one forward pass')
    parser.add_argument('-attack_gnia_homophily', type=float, default=0.5, help='homophily regularization weight for GNIA')
    parser.add_argument('-attack_sequential_step', type=float, default=0.2, help='fraction of nodes injected per sequential attack round')
    parser.add_argument('-attack_quiet', action='store_true', default=False, help='disable per-step attack logs')

    args = parser.parse_args()
    main(args)