import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from tqdm import tqdm, trange
from utils import accuracy, count_arr, split_train_mask
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
# torch.multiprocessing.set_start_method("spawn")


def train_smoothing_model_nc(model, features, adj, labels, idx_train, idx_val, idx_test, optimizer, args):
    endure_count = 0
    best_acc_val = 0

    no_dense = True if args.dataset == 'Amazon' else False
    separate_Amazon = True

    print(f'Training the model for node classification:')
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if separate_Amazon and args.dataset == 'Amazon':
            num_nodes = features.shape[0]
            #train_subs = split_indices(num_nodes)
            sub_masks = split_train_mask(idx_train, split_ratio=0.05, num_splits=5)
            loss_train = torch.zeros(1).to(args.device)
            for sub_masks in sub_masks:
                sub_edge_index, _ = subgraph(
                    sub_masks, 
                    adj, 
                    num_nodes=num_nodes, 
                    relabel_nodes=False
                )
                sub_out = model.forward_perturb(features, sub_edge_index, no_dense=no_dense)
                if args.criterion == 'nll':
                    loss_train += F.nll_loss(sub_out[idx_train], labels[idx_train])
                elif args.criterion == 'ce':
                    loss_train += F.cross_entropy(sub_out[idx_train], labels[idx_train])
                else:
                    raise Exception(f"unknown criterion: {args.criterion}")
                
            with torch.no_grad():
                output = model.forward_perturb(features, adj, no_dense=no_dense)
        else:
            output = model.forward_perturb(features, adj, no_dense=no_dense)
            if args.criterion == 'nll':
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            elif args.criterion == 'ce':
                loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            else:
                raise Exception(f"unknown criterion: {args.criterion}")
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max)
        optimizer.step()
        # evaluate validation set performance
        if epoch %10==0:
            model.eval()
            output = model.forward_perturb(features, adj, no_dense=no_dense)
            if args.criterion == 'nll':
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            elif args.criterion == 'ce':
                loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            if  acc_val > best_acc_val:
                best_acc_val = acc_val
                endure_count = 0
                if args.save_model==True:
                    torch.save(model, args.model_dir)
                    print('model saved to:',args.model_dir)
            else:
                endure_count += 1
            if endure_count > args.patience and best_acc_val>0.99:
                print('early stop at epoch:',epoch)
                break


def train_smoothing_model_gc_old(model, graphs, idx_train, idx_val, idx_test, optimizer, args):
    '''
     This is the old version of train_smoothing_model_gc, which does not use dataloader and probably has some issues.'''
    endure_count = 0
    best_acc_val = 0
    NUM_SUB_EPOCHS = 3

    print(f'Training the model for graph classification:')
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if args.criterion == 'nll':
            criterion = torch.nn.NLLLoss()
        elif args.criterion == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception(f"unknown criterion: {args.criterion}")
        loss_train = 0.0
        len_train = len(idx_train)
        outputs = []
        labels = []
        for i in range(NUM_SUB_EPOCHS):  # for each epoch, do 10 forward passes, 1 for clean graph, 9 for perturbed graphs
            batch_loss = torch.zeros(1).to(args.device)
            count = 0
            for j, graph in enumerate(graphs[idx_train]):
                '''if i == 0:
                    output = model.forward(graph.x.to(args.device), graph.edge_index.to(args.device))
                else:
                    output = model.forward_perturb(graph.x.to(args.device), graph.edge_index.to(args.device), train=True)'''
                output = model.forward_perturb(graph.x.to(args.device), graph.edge_index.to(args.device), train=True)
                outputs.append(output.detach().clone())
                labels.append(graph.y.detach().clone().to(device=args.device, dtype=torch.long))
                # to sum up the loss over all training graphs
                loss_train += criterion(output, graph.y.to(device=args.device, dtype=torch.long)).item()
                # for backpropagation
                batch_loss += criterion(output, graph.y.to(device=args.device, dtype=torch.long))

                j_ = j + 1
                count += 1
                if j_ % 64 == 0 or j_ == len_train:
                    batch_loss = batch_loss / count
                    count = 0
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max)
                    optimizer.step()
                    batch_loss = torch.zeros(1).to(args.device)

        loss_train = loss_train / (len(idx_train) * NUM_SUB_EPOCHS)
            
        acc_train = accuracy(torch.cat(outputs, dim=0), torch.cat(labels, dim=0))
        #loss_train.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max)
        #optimizer.step()

        # evaluate validation set performance
        if epoch %10==0:
            model.eval()
            loss_val = torch.zeros(1).to(args.device)
            outputs = []
            labels = []
            for i, graph in enumerate(graphs[idx_val]):
                output = model.forward_perturb(graph.x.to(args.device), graph.edge_index.to(args.device))
                outputs.append(output)
                labels.append(graph.y.to(device=args.device, dtype=torch.long))
                if args.criterion == 'nll':
                    loss_val += F.nll_loss(output, graph.y.to(device=args.device, dtype=torch.long))
                elif args.criterion == 'ce':
                    loss_val += F.cross_entropy(output, graph.y.to(device=args.device, dtype=torch.long))
                else:
                    raise Exception(f"unknown criterion: {args.criterion}")

            loss_val = loss_val / len(idx_val)
            acc_val = accuracy(torch.cat(outputs, dim=0), torch.cat(labels, dim=0))
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            if  acc_val > best_acc_val:
                best_acc_val = acc_val
                endure_count = 0
                if args.save_model==True:
                    torch.save(model, args.model_dir)
                    print('model saved to:',args.model_dir)
            else:
                endure_count += 1
            if endure_count > args.patience:
                print('early stop at epoch:',epoch)
                break


def train_smoothing_model_gc(model, train_loader, val_loader, optimizer, args):
    endure_count = 0
    best_acc_val = 0
    best_loss_val = None
    NUM_SUB_EPOCHS = 3
    num_eval_repeats = getattr(args, 'eval_repeats', 5)
    model_dir = os.path.dirname(args.model_dir)
    train_csv_path = os.path.join(model_dir, 'train.csv')
    train_result_path = os.path.join(model_dir, 'train_result.txt')

    with open(train_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss_train', 'acc_train', 'loss_val', 'acc_val', 'time'])

    def evaluate_graph_loader(eval_model, data_loader, repeats=None):
        if repeats is None:
            repeats = num_eval_repeats

        eval_model.eval()
        loss_values = []
        acc_values = []

        with torch.no_grad():
            for _ in range(repeats):
                loss_eval = 0.0
                outputs_eval = []
                labels_eval = []

                for data in data_loader:
                    data = data.to(args.device)
                    output = eval_model.forward_perturb(data.x, data.edge_index, data.batch)
                    outputs_eval.append(output)
                    labels_eval.append(data.y.to(device=args.device, dtype=torch.long))
                    if args.criterion == 'nll':
                        loss_eval += F.nll_loss(output, data.y.to(device=args.device, dtype=torch.long)).item() * data.num_graphs
                    elif args.criterion == 'ce':
                        loss_eval += F.cross_entropy(output, data.y.to(device=args.device, dtype=torch.long)).item() * data.num_graphs
                    else:
                        raise Exception(f"unknown criterion: {args.criterion}")

                loss_eval = loss_eval / len(data_loader.dataset)
                acc_eval = accuracy(torch.cat(outputs_eval, dim=0), torch.cat(labels_eval, dim=0)).item()
                loss_values.append(loss_eval)
                acc_values.append(acc_eval)

        return float(np.mean(loss_values)), float(np.mean(acc_values))

    print(f'Training the model for graph classification:')
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if args.criterion == 'nll':
            criterion = torch.nn.NLLLoss()
        elif args.criterion == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception(f"unknown criterion: {args.criterion}")
        loss_train = 0.0
        len_train = len(train_loader.dataset)
        outputs = []
        labels = []

        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            output = model.forward_perturb(data.x, data.edge_index, data.batch, train=True)
            outputs.append(output.detach().clone())
            labels.append(data.y.detach().clone().to(device=args.device, dtype=torch.long))
            loss = criterion(output, data.y.to(device=args.device, dtype=torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max)
            optimizer.step()
            loss_train += loss.item() * data.num_graphs

        loss_train = loss_train / len_train

        acc_train = accuracy(torch.cat(outputs, dim=0), torch.cat(labels, dim=0))
        #loss_train.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max)
        #optimizer.step()

        # evaluate validation set performance
        if epoch %10==0:
            loss_val, acc_val = evaluate_graph_loader(model, val_loader)
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val),
                  'time: {:.4f}s'.format(time.time() - t))
            with open(train_csv_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    epoch + 1,
                    loss_train,
                    acc_train.item(),
                    loss_val,
                    acc_val,
                    time.time() - t,
                ])
            if  acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                endure_count = 0
                if args.save_model==True:
                    torch.save(model, args.model_dir)
                    print('model saved to:',args.model_dir)
            else:
                endure_count += 1
            if endure_count > args.patience:
                print('early stop at epoch:',epoch)
                break

    best_model = model
    if os.path.exists(args.model_dir):
        best_model = torch.load(args.model_dir)
        best_model.to(args.device)
    loss_final, acc_final = evaluate_graph_loader(best_model, val_loader)
    with open(train_result_path, 'w') as txt_file:
        txt_file.write(f'eval_repeats: {num_eval_repeats}\n')
        txt_file.write(f'final_loss_val: {loss_final:.6f}\n')
        txt_file.write(f'final_acc_val: {acc_final:.6f}\n')
        if best_loss_val is not None:
            txt_file.write(f'best_loss_val: {best_loss_val:.6f}\n')
        txt_file.write(f'best_acc_val: {best_acc_val:.6f}\n')


def train_N_models(model,dataset,optimizer,args,save_dir_inc,save_dir_exc):
    best_acc_val = 0
    adj, features, labels, idx_train, idx_val, idx_test = dataset
    n=features.shape[0]
    adj_dense = torch.squeeze(to_dense_adj(adj))
    counts_inc = np.zeros((features.shape[0], model.nclass), dtype=int)
    counts_exc = np.zeros((features.shape[0], model.nclass), dtype=int)

    print('Preparaing smoothing samples:')
    Data_list=[]
    for repeat_time in tqdm(range(int(args.n_smoothing))):
        # sampling the smoothing distribution.
        adj_dense_pert = model.perturbation(adj_dense)
        # delete the isolated/singleton nodes in training set
        deleted_nodes = torch.where(torch.sum(adj_dense_pert, dim=0) == 0)[0].cpu().numpy().tolist()
        keep_index = set(range(n)) - set(deleted_nodes)
        sub_train_index = torch.tensor(list(keep_index.intersection(idx_train)))
        edge_index_pert = torch.nonzero(adj_dense_pert).t()
        if sub_train_index.shape[0]>0:
            Data_list.append(
                Data(
                    x=features, 
                    edge_index=edge_index_pert,
                    train_index=sub_train_index,
                    keep_index=list(keep_index),
                    idx_val=idx_val
                )
            )
        else: # do it against
            # sampling the smoothing distribution.
            adj_dense_pert = model.perturbation(adj_dense)
            # delete the isolated/singleton nodes in training set
            deleted_nodes = torch.where(torch.sum(adj_dense_pert, dim=0) == 0)[0].cpu().numpy().tolist()
            keep_index = set(range(n)) - set(deleted_nodes)
            sub_train_index = torch.tensor(list(keep_index.intersection(idx_train)))
            edge_index_pert = torch.nonzero(adj_dense_pert).t()
            if sub_train_index.shape[0] > 0:
                Data_list.append(Data(x=features, edge_index=edge_index_pert, train_index=sub_train_index,
                                      keep_index=list(keep_index), idx_val=idx_val))

    # dataloader
    batch_loader = DataLoader(Data_list, batch_size=1, shuffle=False)
    for i,batch in enumerate(tqdm(batch_loader)):
        # train the model
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model.forward(batch.x, batch.edge_index)
            loss_train = F.nll_loss(output[batch.train_index], labels[batch.train_index])
            acc_train = accuracy(output[batch.train_index], labels[batch.train_index])
            loss_train.backward()
            optimizer.step()

            if i<10 and epoch % 99 == 0 and epoch>0:
                model.eval()
                output = model.forward(batch.x, batch.edge_index)
                loss_val = F.nll_loss(output[batch.idx_val], labels[batch.idx_val])
                acc_val = accuracy(output[batch.idx_val], labels[batch.idx_val])
                print('Model: {:01d}'.format(i + 1),
                      'epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))


        predictions = model.forward(batch.x, batch.edge_index).argmax(1)
        # if args.singleton == 'exclude':
        predictions_exc=predictions[batch.keep_index]
        counts_exc[batch.keep_index,:] += count_arr(predictions_exc.cpu().numpy(), model.nclass)
        # if args.singleton == 'include':
        counts_inc += count_arr(predictions.cpu().numpy(), model.nclass)
        model.reset_parameters()
        #! AttributeError: 'SmoothGAT' object has no attribute 'reset_parameters'. Did you mean: 'get_parameter'?

    top2_exc = counts_exc.argsort()[:, ::-1][:, :2].copy()
    count1_exc = [counts_exc[n, idx] for n, idx in enumerate(top2_exc[:, 0])]
    count2_exc = [counts_exc[n, idx] for n, idx in enumerate(top2_exc[:, 1])]
    f = open(save_dir_exc, 'wb')
    pickle.dump([top2_exc, count1_exc, count2_exc, counts_exc], f)
    f.close()
    print(f'Save result to {save_dir_exc}')

    top2_inc = counts_inc.argsort()[:, ::-1][:, :2].copy()
    count1_inc = [counts_inc[n, idx] for n, idx in enumerate(top2_inc[:, 0])]
    count2_inc = [counts_inc[n, idx] for n, idx in enumerate(top2_inc[:, 1])]

    f = open(save_dir_inc, 'wb')
    pickle.dump([top2_inc, count1_inc, count2_inc, counts_inc], f)
    f.close()
    print(f'Save result to {save_dir_inc}')

    if args.singleton == 'exclude':
        return top2_exc, count1_exc, count2_exc, counts_exc
    else:
        return top2_inc, count1_inc, count2_inc, counts_inc


