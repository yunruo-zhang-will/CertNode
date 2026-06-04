import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import time
from tqdm import tqdm, trange
import concurrent.futures
from datetime import datetime
import csv

pp = pprint.PrettyPrinter(depth=4)

import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ncx2

from utils import *
from models import *
from train import *
from certify import *
from attacks.tdgia_eot import EOTTDGIAAttack
from attacks.gnia_eot import EOTGNIAAttack
from attacks.grb_injection_baseline import EOTGRBInjectionAttack
from attacks.injection_common import evaluate_smoothed_attack

import warnings
warnings.filterwarnings("ignore")

import argparse
import matplotlib as mpl
from matplotlib import rc
from torch_geometric.loader import DataLoader


MEDIUM_SIZE = 25
BIGGER_SIZE = 27

def init_random_seed(SEED=2021):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
    warnings.filterwarnings("ignore")

def certify_process(rho, top2, count1, count2, sample_config, nclass, args, idx_test, correct):
    cAHat, certified = certify(
        rho,
        top2[idx_test],
        listSubset(count1, idx_test),
        listSubset(count2, idx_test),
        sample_config,
        nclass,
        args
    )
    certified_and_correct = np.array(certified) & correct
    certified_accuracy = np.sum(certified_and_correct) / len(idx_test)
    return certified_accuracy, cAHat, certified


def save_attack_results(output_dir, attack_eval, attack_meta, attack_info, attacked_features, attacked_edge_index, attack_seconds, args):
    attack_result_path = os.path.join(output_dir, 'attack_result.txt')
    if os.path.exists(attack_result_path):
        os.remove(attack_result_path)

    attack_info_path = os.path.join(output_dir, 'attack_info.txt')
    if os.path.exists(attack_info_path):
        os.remove(attack_info_path)

    torch.save(
        {
            'attack_method': args.attack_method,
            'features': attacked_features.detach().cpu(),
            'edge_index': attacked_edge_index.detach().cpu(),
            'injected_nodes': attack_meta.injected_nodes.detach().cpu(),
            'target_index': attack_meta.target_index.detach().cpu(),
            'attacked_top1': attack_eval.attacked_top1,
            'attacked_top2': attack_eval.attacked_top2,
            'attacked_count1': attack_eval.attacked_count1,
            'attacked_count2': attack_eval.attacked_count2,
        },
        os.path.join(output_dir, 'attacked_graph.pt')
    )

    csv_path = os.path.join(output_dir, 'attack_result.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['metric', 'value'])
        csv_writer.writerow(['attack_method', args.attack_method])
        csv_writer.writerow(['clean_accuracy', attack_eval.clean_accuracy])
        csv_writer.writerow(['attacked_accuracy', attack_eval.attacked_accuracy])
        csv_writer.writerow(['attack_success_rate', attack_eval.asr])
        csv_writer.writerow(['attack_seconds', attack_seconds])
        csv_writer.writerow(['attack_minutes', attack_seconds / 60])
        csv_writer.writerow(['eval_count', attack_eval.eval_count])
        csv_writer.writerow(['clean_correct_count', attack_eval.clean_correct_count])
        csv_writer.writerow(['injected_node_count', len(attack_meta.injected_nodes)])
        csv_writer.writerow(['attack_target_count', attack_meta.target_count])
        csv_writer.writerow(['attack_clean_smoothing', args.attack_clean_smoothing])
        csv_writer.writerow(['attack_eval_smoothing', args.attack_eval_smoothing])
        csv_writer.writerow(['attack_rank_samples', args.attack_rank_samples])
        csv_writer.writerow(['attack_opt_samples', args.attack_opt_samples])
        csv_writer.writerow(['attack_gnia_homophily', args.attack_gnia_homophily])
        csv_writer.writerow(['ranking_rounds', attack_info.ranking_rounds])
        csv_writer.writerow(['optimization_epochs', attack_info.optimization_epochs])
        csv_writer.writerow(['sequential_rounds', attack_info.sequential_rounds])
        csv_writer.writerow(['theoretical_model_calls', attack_info.theoretical_model_calls])

def main(args):

    rc('text', usetex=True)
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
    plt.style.use('classic')

    # codes for making the plots
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

    # Print the device
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpuID}')
        print(f"---using GPU---cuda:{args.gpuID}----")
    else:
        print("---using CPU---")
        args.device = torch.device("cpu")

    init_random_seed(args.seed)

    # Load dataset node classification and graph classification use different datasets
    from dataset_loader import load_node_data, load_graph_data
    if args.task == 'Node':
        if args.dataset == 'Amazon':
            from ogb.nodeproppred import PygNodePropPredDataset
            datasets = PygNodePropPredDataset(name = "ogbn-products") 
            split_idx = datasets.get_idx_split()
            graph = datasets[0] 
            features = torch.tensor(graph.x).to(args.device)
            edge_index = torch.tensor(graph.edge_index).to(args.device)
            labels = torch.tensor(graph.y).squeeze().to(args.device)
            num_x = features.shape[1]
            num_labels = 47
            #print(f'graph.x type: {type(graph.x)}')
            #print(f'graph.x shape: {graph.x.shape}')
            #print(f'graph.edge_index shape: {graph.edge_index.shape}')
            #print(f'edge_index shape: {edge_index.shape}')
            #print(f'label shape: {labels.shape}')
            #exit(0)
            # big dataset
            # 2,449,029 node, each 100 d feature
            # 61,859,140 edge, edge_index shape: torch.Size([2, 123718280]) (two direction for one edge)

            train_mask = []  # list of tensor indices (shape: [])
            val_mask = []
            test_mask = []
            # make sure the ratio of train:val:test is similar among all classes
            for c in range(num_labels):
                idx = (torch.tensor(labels) == c).nonzero(as_tuple=False).reshape(-1)
                train_i = idx[:int(0.3*len(idx))]
                val_i = idx[int(0.3*len(idx)):int(0.5*len(idx))]
                test_i = idx[int(0.5*len(idx)):-1]
                train_mask.extend(train_i.cpu())
                val_mask.extend(val_i.cpu())
                test_mask.extend(test_i.cpu())

            '''print(f'train_mask type: {type(train_mask)}')
            print(f'train_mask len: {len(train_mask)}')
            print(f'train_mask[0] type: {type(train_mask[0])}')
            print(f'train_mask[0] shape: {train_mask[0].shape}')
            print(f'train_mask[0]: {train_mask[0]}')
            exit(0)'''
            '''idx_test = get_mask_indices(test_mask)
            print(f'type of idx_test: {type(idx_test)}')  # numpy.ndarray
            print(f'number of test nodes: {len(idx_test)}')
            exit(0)'''
        else:
            if args.dataset == "PubMed":
                num_train=2000
                num_val = 600
            elif args.dataset == "computers":
                num_train=400
                num_val = 133
            else:
                num_train= 150
                num_val = 50
            data, num_x, num_labels = load_node_data(args.dataset, num_train=num_train, num_val=num_val) # num_node_features, num_classes
            features = torch.tensor(data.x).to(args.device)
            edge_index = torch.tensor(data.edge_index).to(args.device)
            labels = torch.tensor(data.y).to(args.device)
            train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    elif args.task == 'Graph':
        if args.dataset == "Mutagenicity":
            num_train = 1000
            num_val=400
        else:
            num_train = 250
            num_val=100
        graphs, num_x, num_labels, mask_spilt, labels = load_graph_data(args.dataset, num_train=num_train, num_val=num_val)
        train_mask = mask_spilt[0]
        val_mask = mask_spilt[1]
        test_mask = mask_spilt[2]
        # print(f'type(train_mask): {type(train_mask)}, train_mask.shape: {train_mask.shape}')
        # type(train_mask): <class 'torch.Tensor'>, train_mask.shape: torch.Size([1113])
        # exit(0)

        # 根据 mask 提取出对应的数据子集
        train_dataset = graphs[train_mask]
        val_dataset = graphs[val_mask]
        test_dataset = graphs[test_mask]
        # 4. 创建 DataLoader（这一步会自动为你生成 batch 变量！）
        # batch_size 可以设置为 32 或 64
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    else:
        raise Exception(f"unknown task: {args.task}")

    # Smoothing samples config
    sample_config = {'p_n': args.p_n}
    #args.output_dir = f'{args.output_dir}/{args.dataset}_{args.model}_{sample_config["p_n"]}_seed{args.seed}_{args.n_smoothing}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    #args.model_dir = f'{args.output_dir}/{args.model}.pth'
    if False: #args.seed == 2020:  # default seed, do not save seed in dir
        model_dir = f'./models/{args.dataset}_{args.model}_{sample_config["p_n"]}'
        args.output_dir = f'{args.output_dir}/{args.dataset}_{args.model}_{sample_config["p_n"]}_{args.n_smoothing}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        model_dir = f'./models/{args.dataset}_{args.model}_{sample_config["p_n"]}_seed{args.seed}'  
        output_prefix = f'{args.attack_method}_{args.dataset}' if args.run_attack else args.dataset
        args.output_dir = f'{args.output_dir}/{output_prefix}_{args.model}_{sample_config["p_n"]}_seed{args.seed}_{args.n_smoothing}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    args.model_dir = f'{model_dir}/model.pth'  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    ## Load model
    #from models import SmoothNodeGCN, SmoothNodeGAT, SmoothNodeGSAGE, SmoothGraphGCN, SmoothGraphGAT, SmoothGraphGSAGE
    if args.task == 'Node':
        if args.model == "GCN":
            model = SmoothNodeGCN(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        elif args.model == "GAT":
            model = SmoothNodeGAT(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        elif args.model == "GSAGE":
            model = SmoothNodeGSAGE(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        elif args.model == "GIN":
            model = SmoothNodeGIN(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        elif args.model == "APPNP":
            model = SmoothNodeAPPNP(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        else:
            raise Exception(f"unknown model: {args.model}")
    elif args.task == 'Graph':
        if args.model == "GCN":
            model = SmoothGraphGCN(num_x, num_labels, args.n_hidden, dropout=args.drop, config=sample_config, device=args.device).to(args.device)
        elif args.model == "GAT":
            model = SmoothGraphGAT(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        elif args.model == "GSAGE":
            model = SmoothGraphGSAGE(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        elif args.model == "GIN":
            model = SmoothGraphGIN(num_x, num_labels, args.n_hidden, dropout=args.drop, config=sample_config, device=args.device).to(args.device)
        elif args.model == "APPNP":
            model = SmoothGraphAPPNP(num_x, num_labels, args.n_hidden, config=sample_config, device=args.device).to(args.device)
        else:
            raise Exception(f"unknown model: {args.model}")
    else:
        raise Exception(f"unknown task: {args.task}")
    
    print(f'Model: {args.model} on {args.device}.')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ## Training the model with smoothing perturbation samples
    if not os.path.exists(args.model_dir) or args.force_training:
        # measure training time for reporting/diagnostics
        t_start = time.time()
        if args.task == 'Node':
            train_smoothing_model_nc(model, features, edge_index, labels, train_mask, val_mask, test_mask, optimizer, args)
        elif args.task == 'Graph':
            train_smoothing_model_gc(model, train_loader, val_loader, optimizer, args)
        t_end = time.time()
        elapsed = t_end - t_start
        print(f"Training finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        # also persist the timing to output dir
        try:
            with open(os.path.join(args.output_dir, 'training_time.txt'), 'w') as tf:
                tf.write(f"training_seconds: {elapsed:.6f}\n")
                tf.write(f"training_minutes: {elapsed/60:.6f}\n")
        except Exception as e:
            print(f"Failed to write training time: {e}")

    if os.path.exists(args.model_dir):
        model = torch.load(args.model_dir)
        model.to(args.device)

    node_no_dense = args.task == 'Node' and args.dataset == 'Amazon'

    # Smoothing Testing
    if not os.path.exists(f'{args.output_dir}/smoothing_result.pkl') or args.force_training:
        model.eval()
        if args.task == 'Node':
            tic = time.time()
            top2, count1, count2 = model.smoothed_precit(features, edge_index, num=args.n_smoothing, no_dense=node_no_dense)
            toc = time.time()
            inference_time = toc - tic
            print(f'Smoothed inference time for all nodes: {inference_time:.2f} seconds')
            try:
                with open(os.path.join(args.output_dir, 'inference_time.txt'), 'w') as tf:
                    tf.write(f"inference_seconds: {inference_time:.6f}\n")
                    tf.write(f"inference_minutes: {inference_time/60:.6f}\n")
            except Exception as e:
                print(f"Failed to write training time: {e}")
            # print(f'top2.type: {type(top2)}, count1.type: {type(count1)}, count2.type: {type(count2)}')
            # top2.type: <class 'numpy.ndarray'>, count1.type: <class 'list'>, count2.type: <class 'list'>
            # print(f'top2.shape: {top2.shape}, count1.shape: {len(count1)}, count2.shape: {len(count2)}')
            # top2.shape: (3327, 2), count1.shape: 3327, count2.shape: 3327
            # print(f'count1[0]: {count1[0]}')
            # count1[0]: 10000
            # exit(0)
        elif args.task == 'Graph':
            #TODO check if top2, count1, count2 are correctly collected
            top2, count1, count2 = [], [], []
            csv_file_path = f'{args.output_dir}/smoothing_result.csv'
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['top2', 'count1', 'count2'])  # Write the header row
            
            for data in test_loader:
                data = data.to(args.device)
                t2, c1, c2 = model.smoothed_precit(data.x, data.edge_index, data.batch, num=args.n_smoothing)
                top2 = t2 if len(top2) == 0 else torch.cat((top2, t2), dim=0)
                count1 = count1 + c1
                count2 = count2 + c2

                #for i, graph in enumerate(graphs):
                #print(f'Smoothed inference graph {i}:')
                #t2, c1, c2 = model.smoothed_precit(graph.x.to(args.device), graph.edge_index.to(args.device), num=args.n_smoothing)
                #print(f't2 type: {type(t2)}, t2 shape: {t2.shape}')
                #print(f'c1 type: {type(c1)}, c1 shape: {len(c1)}')
                #print(f'c2 type: {type(c2)}, c2 shape: {len(c2)}')
                #exit(0)
                #top2.append([t2])
                #count1 += c1
                #count2 += c2
                print('\033[2A\033[0J', end='')  # Clear the previous two lines
                #print(f't2: {t2}')
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(zip(t2[:, 0].tolist(), c1, c2))
                '''if i % 5 == 0:
                    print(f'number of graphs: {len(graphs)}')
                    print(f'count1 type: {type(count1)}, count1[0] type: {type(count1[0])}')
                    print(f'count1 shape: {len(count1)}')
                    print(f'count2 type: {type(count2)}, count2[0] type: {type(count2[0])}')
                    print(f'count2 shape: {len(count2)}')'''
                #! test, use 100 graphs 
                '''if i == 99:
                    break'''
            print('Graph smoothed inference done.')
            top2 = top2.cpu().numpy()
            '''print(f'top2 type: {type(top2)}, top2 shape: {top2.shape}')
            print(f'count1 type: {type(count1)}, count1 shape: {len(count1)}')
            print(f'count2 type: {type(count2)}, count2 shape: {len(count2)}')
            exit(0)'''
            #print(f'number of graphs: {len(graphs)}')
            #print(f'count1 type: {type(count1)}, count1[0] type: {type(count1[0])}')
            #print(f'count1 shape: {len(count1)}, count1[0] shape: {len(count1[0])}')
            #print(f'count2 type: {type(count2)}, count2[0] type: {type(count2[0])}')
            #print(f'count2 shape: {len(count2)}, count2[0] shape: {len(count2[0])}')
            '''
            number of graphs: 1113
            count1 type: <class 'list'>, count1[0] type: <class 'list'>
            count1 shape: 1113, count1[0] shape: 42
            count2 type: <class 'list'>, count2[0] type: <class 'list'>
            count2 shape: 1113, count2[0] shape: 42
            '''
            #exit(0)
            print(f'Save result to {csv_file_path}')
        f = open(f'{args.output_dir}/smoothing_result.pkl', 'wb')
        pickle.dump([top2, count1, count2], f)
        f.close()
        print(f'Save result to {args.output_dir}/smoothing_result.pkl')

    else:
        # add task choice below
        f = open(f'{args.output_dir}/smoothing_result.pkl', 'rb')
        top2, count1, count2 = pickle.load(f)
        f.close()

    if args.task == 'Node':
        idx_test = get_mask_indices(test_mask)
        test_label = labels.cpu().numpy()[test_mask]
        correct = (np.array(top2[idx_test, 0]) == test_label)
        print("Smoothed classifier accuracy:", np.sum(correct) / len(idx_test))
    elif args.task == 'Graph':
        test_label = []
        for i, graph in enumerate(graphs[test_mask]):
            test_label.append(graph.y.item())
        test_label = np.array(test_label)
        idx_test = np.arange(len(test_label))
        # ValueError: operands could not be broadcast together with shapes (411,) (200,)
        correct = (np.array(top2[:, 0]) == test_label)
        print("Smoothed classifier accuracy:", np.sum(correct) / len(idx_test))
    else:
        raise Exception(f"unknown task: {args.task}")

    if args.run_attack:
        raise Exception('This file is no longer used for attacks.')
        if args.task != 'Node':
            raise Exception('Node injection attacks are only implemented for Node task.')

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
        test_mask_device = test_mask.to(args.device)
        attack_target_mask[test_mask_device] = clean_top1[test_mask_device] == labels[test_mask_device]

        attack_kwargs = dict(
            n_inject_max=args.attack_n_inject_max,
            n_edge_max=args.attack_n_edge_max,
            feat_lim_min=float(features.min().item()),
            feat_lim_max=float(features.max().item()),
            lr=args.attack_lr,
            n_epoch=args.attack_epochs,
            sequential_step=args.attack_sequential_step,
            n_rank_samples=args.attack_rank_samples,
            n_opt_samples=args.attack_opt_samples,
            device=args.device,
            verbose=not args.attack_quiet,
        )
        if args.attack_method == 'tdgia':
            attack = EOTTDGIAAttack(**attack_kwargs)
        elif args.attack_method == 'gnia':
            attack = EOTGNIAAttack(**attack_kwargs, homophily_weight=args.attack_gnia_homophily)
        elif args.attack_method == 'grb_injection':
            attack = EOTGRBInjectionAttack(**attack_kwargs)
        else:
            raise Exception(f"unknown attack method: {args.attack_method}")

        attack_start = time.time()
        attacked_features, attacked_edge_index, attack_meta, attack_info = attack.attack(
            model=model,
            features=features,
            edge_index=edge_index,
            anchor_labels=clean_top1,
            target_mask=attack_target_mask,
            no_dense=node_no_dense,
        )
        attack_end = time.time()
        print(f'{args.attack_method} attack finished in {attack_end - attack_start:.2f} seconds')

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

        print(f"Attacked smoothed accuracy: {attack_eval.attacked_accuracy:.6f}")
        print(f"Attack success rate: {attack_eval.asr:.6f}")
        save_attack_results(
            args.output_dir,
            attack_eval,
            attack_meta,
            attack_info,
            attacked_features,
            attacked_edge_index,
            attack_end - attack_start,
            args,
        )
        print('Attack-only mode finished. Skip certify computation.')
        return attack_eval

    # Certify
    def run(certify_process, rho_list):
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(certify_process, rho_list), total=len(rho_list)))
        return results

    max_rho = args.max_rho
    rho_list = [*range(1, max_rho + 1, 1)]
    certi_acc_list = []

    if False:  # args.p_e == 0 or args.p_n == 0: # single process
        zero_cer = False
        for rho in tqdm(rho_list, desc='Processing certified radius'):
            if not zero_cer:
                certified_accuracy, cAHat, _ = certify_process(rho)
                certi_acc_list.append(certified_accuracy)
                if certified_accuracy == 0.0:
                    zero_cer = True  # Then, do not need further verify for larger radius.
            else:
                certi_acc_list.append(0.0)
    else:  # multi-thread
        # rho_list = [*range(0, max_rho + 1, 20)][1:]
        print('multi-process mode')
        #results = run(certify_process, rho_list)
        from functools import partial
        certify_func = partial(
            certify_process, top2=top2, count1=count1, count2=count2,
            sample_config=sample_config, nclass=model.nclass, args=args,
            idx_test=idx_test, correct=correct
        )
        results = run(certify_func, rho_list)

        for res in results:
            certi_acc_list.append(res[0])
            cAHat = res[1]

    print('The ratio of ABSTAIN:', np.sum([True if ca == -1 else False for ca in cAHat]) / len(cAHat))

    # Save the certified radius and accuracy data to a pickle file
    df = {'rho': [0] + rho_list, 'certified accuracy': [np.sum(correct) / len(idx_test)] + certi_acc_list}
    f = open(f'{args.output_dir}/certify_result.pkl', 'wb')
    pickle.dump(df, f)
    f.close()
    print(f'Save result to {args.output_dir}/certify_result.pkl')

    plt.figure(constrained_layout=True)
    sns.lineplot(x="rho", y="certified accuracy", data=df)
    plt.xlabel(r'$\rho$')
    plt.title(fr"$p_n$={sample_config['p_n']}")
    plt.savefig(args.output_dir + f'/{sample_config["p_n"]}_{args.n_smoothing}_certify_curve.pdf', dpi=300)
    print(f'Save result to {args.output_dir}{sample_config["p_n"]}{args.n_smoothing}_certify_curve.pdf')
    plt.show()

    # Save the certified radius and accuracy data to a CSV file
    csv_file_path = f'{args.output_dir}/certify_result.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['rho', 'certified_accuracy'])  # Write the header row
        for rho, accuracy in zip([0] + rho_list, [np.sum(correct) / len(idx_test)] + certi_acc_list):
            csv_writer.writerow([rho, accuracy])
    print(f'Save result to {csv_file_path}')

    k_list = [0] + rho_list
    acc_list = [np.sum(correct) / len(idx_test)] + certi_acc_list
    acc_list = [float(acc) for acc in acc_list]

    auc = sum(acc_list) - (acc_list[0] / 2) - (acc_list[-1] / 2)

    txt_file_path = f'{args.output_dir}/certify_result.txt'
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f'k_list: \n{k_list}')
        txt_file.write(f'\n\nacc_list: \n{acc_list}')
        txt_file.write(f'\n\nAUC: \n{auc}')
    print(f'Save result to {txt_file_path}')

    return k_list, acc_list


if __name__ == "__main__":

    # Model Settings=======================================
    parser = argparse.ArgumentParser(description='certify GNN node injecttion')

    parser.add_argument('-gpuID', type=int, default=0)
    parser.add_argument('-seed', type=int, default=2020)
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-clip_max', type=float, default=2.0, help='gradient clipping max norm')
    parser.add_argument('-criterion', type=str, default='nll', choices=['nll', 'ce'], help='loss function')
    parser.add_argument('-patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('-epochs', type=int, default=1000, help='training epoch')
    parser.add_argument('-save_model', action='store_true', default=True, help="save model")
    parser.add_argument('-task', type=str, default='None', choices=['Node', 'Graph'], 
                        help='GNN tasks, Node for node classification, Graph for graph classification')
    parser.add_argument('-dataset', type=str, default='CiteSeer', 
                        choices=['PubMed', 'CiteSeer', 'computers', 'Cora-ML', 'Amazon', # for node classification, computers is Amazon-C, Amazon is Amazon2M
                                'Mutagenicity', 'PROTEINS', 'DD', 'AIDS'])    # for graph classification
    parser.add_argument('-model', type=str, default='GAT', choices=['GCN', 'GAT', 'GSAGE', 'GIN', 'APPNP'], help='GNN model')
    parser.add_argument('-n_hidden', type=int, default=64, help='size of hidden layer')
    parser.add_argument('-drop', type=float, default=0.1, help='dropout rate')
    parser.add_argument('-weight_decay', type=float, default=0, help='weight_decay rate') # 1e-3
    parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
    parser.add_argument('-force_training', action='store_true', default=False, 
                        help="force training even if pretrained model exist")

    # Certify setting----------------------
    #parser.add_argument('-certify_mode', type=str, default='evasion', choices=['evasion', 'poisoning'])
    parser.add_argument('-p_n', type=float, default=0.9, help='probability of deleting nodes')
    parser.add_argument('-n_smoothing', type=int, default=10000, help='number of smoothing samples evalute (N)')
    parser.add_argument('-conf_alpha', type=float, default=0.01, help='confident alpha for statistic testing')
    parser.add_argument('-max_rho', type=int, default=200, help='maximum certified radius to compute')

    # Dir setting-------------------------
    parser.add_argument('-output_dir', type=str, default='./results/', help='output directory')

    # Attack setting----------------------
    parser.add_argument('-run_attack', action='store_true', default=False,
                        help='run node injection attack after clean smoothing evaluation')
    parser.add_argument('-attack_method', type=str, default='tdgia',
                        choices=['tdgia', 'gnia', 'grb_injection'],
                        help='node injection attack to run')
    parser.add_argument('-attack_n_inject_max', type=int, default=20,
                        help='maximum number of injected nodes for node injection attacks')
    parser.add_argument('-attack_n_edge_max', type=int, default=5,
                        help='maximum number of edges for each injected node')
    parser.add_argument('-attack_lr', type=float, default=0.01,
                        help='learning rate for injected feature optimization')
    parser.add_argument('-attack_epochs', type=int, default=100,
                        help='optimization epochs for injected node features')
    parser.add_argument('-attack_rank_samples', type=int, default=16,
                        help='number of EOT samples used for topology ranking')
    parser.add_argument('-attack_opt_samples', type=int, default=8,
                        help='number of EOT samples used in each optimization step')
    parser.add_argument('-attack_gnia_homophily', type=float, default=0.5,
                        help='homophily regularization weight for the G-NIA style attack')
    parser.add_argument('-attack_clean_smoothing', type=int, default=10000,
                        help='number of smoothing samples used for clean target selection in attack mode')
    parser.add_argument('-attack_eval_smoothing', type=int, default=10000,
                        help='number of smoothing samples used to evaluate attacked graph')
    parser.add_argument('-attack_sequential_step', type=float, default=0.2,
                        help='fraction of injected nodes added per sequential attack step')
    parser.add_argument('-attack_quiet', action='store_true', default=False,
                        help='disable per-step attack progress logs')

    args = parser.parse_args()

    main(args)



# For node classification
# python main.py -epochs 200 -lr 0.002 -criterion ce -task Node -dataset CiteSeer -model GCN -p_n 0.99 -n_smoothing 10000 -gpuID 5
# CoraML seems bugged
# python main.py -epochs 200 -lr 0.002 -criterion ce -task Node -dataset Cora-ML -model GCN -p_n 0.99 -n_smoothing 10000 -gpuID 5
# test new models GIN and APPNP, both work well
# python main.py -epochs 200 -lr 0.002 -criterion ce -task Node -dataset Cora-ML -model GIN -p_n 0.99 -n_smoothing 10000 -gpuID 5
# python main.py -epochs 200 -lr 0.002 -criterion ce -task Node -dataset Cora-ML -model APPNP -p_n 0.99 -n_smoothing 10000 -gpuID 5
# test Amazon dataset
# python main.py -epochs 200 -lr 0.002 -criterion ce -task Node -dataset Amazon -model GCN -p_n 0.99 -n_smoothing 10000 -gpuID 5
# python main.py -epochs 2000 -lr 0.001 -criterion ce -patience 100 -task Node -dataset Amazon -model GCN -p_n 0.99 -n_smoothing 10000 -gpuID 5


# for graph classification, nan in loss (now debug)
# python main.py -epochs 200 -lr 0.001 -criterion ce -task Graph -dataset PROTEINS -model GCN -p_n 0.99 -n_smoothing 10000 -gpuID 5
# python main.py -epochs 2000 -patience 2000 -lr 0.001 -criterion ce -task Graph -dataset DD -model GCN -p_n 0.99 -n_smoothing 10000 -gpuID 5
# python main.py -epochs 2000 -patience 2000 -lr 0.001 -criterion ce -task Graph -dataset DD -model GIN -p_n 0.99 -n_smoothing 10000 -gpuID 5


