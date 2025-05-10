import os
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import argparse
import pandas as pd

from utils import *
from log.train_logger import TrainLogger
from sklearn.metrics import mean_squared_error
from argparse import Namespace, ArgumentParser, FileType, ArgumentTypeError
import warnings
import pickle
from itertools import product
from tqdm import tqdm
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
from scipy.stats import kendalltau

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_model(model_name, device, model_path=None):
    '''
    model: Literal['MGraphDTA', 'DGraphDTA', 'GraphDTA', 'AttentionDTA', 'DeepDTA']
    '''
    if model_name == 'MGraphDTA':
        from MGraphDTA.model import MGraphDTA
        model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    elif model_name == 'DGraphDTA':
        from DGraphDTA.gnn import GNNNet
        model = GNNNet().to(device)
    elif model_name == 'GraphDTA':
        from GraphDTA.models.gat_gcn import GAT_GCN
        model = GAT_GCN().to(device)
    elif model_name == 'AttentionDTA':
        from AttentionDTA.model import AttentionDTA
        model = AttentionDTA().to(device)
    elif model_name == 'DeepDTA':
        from DeepDTA.model import DeepDTA
        #TODO: add protein and ligand vocab size
        model = DeepDTA(pro_vocab_size=len(train_set.protein_vocab), lig_vocab_size=len(train_set.ligand_vocab), channel=32, protein_kernel_size=12, ligand_kernel_size=8).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_path:
        load_model_dict(model, model_path, device)

    return model

def get_dataset(model_name, root, split_method, split, seed=None, data_df=None):
    '''
    model: Literal['MGraphDTA', 'DGraphDTA', 'GraphDTA', 'AttentionDTA', 'DeepDTA']
    '''
    if model_name == 'MGraphDTA':
        from MGraphDTA.preprocessing import GNNDataset
        dataset = GNNDataset(df=data_df, root=root, split_method=split_method, split=split, seed=seed)
    elif model_name == 'DGraphDTA':
        from DGraphDTA.preprocessing import GNNDataset
        dataset = GNNDataset(df=data_df, root=root, split_method=split_method, split=split, seed=seed)
    elif model_name == 'GraphDTA':
        from GraphDTA.preprocessing import GNNDataset
        dataset = GNNDataset(df=data_df, root=root, split_method=split_method, split=split, seed=seed)
    elif model_name == 'AttentionDTA':
        from AttentionDTA.preprocessing import Dataset
        dataset = Dataset(df=data_df, split_method=split_method, split=split, seed=seed)
    elif model_name == 'DeepDTA':
        from DeepDTA.preprocessing import Dataset
        dataset = Dataset(df=data_df, split_method=split_method, split=split, seed=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return dataset
  

def int_or_false(value):
    if value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        raise ArgumentTypeError(f"Expected int or 'false', got: {value}")

def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def val(model, dataloader, device, model_name):
    model.eval()
    pred_list = []
    label_list = []
    for data in dataloader:
        if model_name == 'DGraphDTA':
            data_protien, data_ligand = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                pred = model(data_protien, data_ligand)
                label = data_protien.y
        elif model_name in ['AttentionDTA', 'DeepDTA']:
            proteins, ligands, label = data
            proteins, ligands, label = proteins.to(device), ligands.to(device), label.to(device)
            with torch.no_grad():
                pred = model(proteins, ligands)
        else:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data)
                label = data.y

        pred_list.append(np.atleast_1d(pred.detach().cpu().numpy()))
        label_list.append(np.atleast_1d(label.detach().cpu().numpy()))
            
    pred = np.concatenate(pred_list, axis=0).flatten()
    label = np.concatenate(label_list, axis=0).flatten()
    coff = np.corrcoef(pred, label)[0, 1]
    cindex = get_cindex(pred, label)
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    

    model.train()

    return mse, rmse, coff, cindex, pred, label


def collate_fn(batch):
    """
    Collate function for the AttentionDTA and DeepDTA DataLoader
    """
    proteins, ligands, affinities = zip(*batch)
    proteins = torch.stack(proteins, dim=0)
    ligands = torch.stack(ligands, dim=0)
    affinities = torch.stack(affinities, dim=0)
    return proteins, ligands, affinities

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model_name', choices=['MGraphDTA', 'DGraphDTA', 'GraphDTA', 'AttentionDTA', 'DeepDTA'], help='model name')
    parser.add_argument('--data_root', type=str, default='data/davis_complete', help='data root')
    parser.add_argument('--data_df', type=str, default='davis_complete.csv', help='data of protein and ligand')
    parser.add_argument('--split_method', choices=['random', 'drug', 'protein', 'both', 'seqid', 'wt_mutation'], help='split method')
    parser.add_argument('--model_seeds', nargs='+', type=int, help='List of seeds for the repeats')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-4)

    args = parser.parse_args()
    device = torch.device(f'cuda:{str(args.gpu)}')
    model_name = args.model_name
    root = os.path.join(model_name, args.data_root)
    data_df = pd.read_csv(args.data_df)
    split_method = args.split_method
    model_seeds = args.model_seeds
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # clear the cache 
    if os.path.exists(os.path.join(root, 'processed')):
        for file in os.listdir(os.path.join(root, 'processed')):
            file_path = os.path.join(root, 'processed', file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    all_test_mse = []
    all_test_rp = []
    all_test_cindex = []
    all_test_wt_mse = []
    all_test_wt_rp = []
    all_test_wt_cindex = []
    all_test_mutation_mse = []
    all_test_mutation_rp = []
    all_test_mutation_cindex = []


    for model_seed in model_seeds:
        if os.path.exists(os.path.join(model_name, 'save', split_method, f'seed_{model_seed}.pt')):
            print(f"Model {model_seed} already exists, skip training")
            continue

        train_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='train', seed=model_seed, data_df=data_df)
        valid_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='valid', seed=model_seed, data_df=data_df)
        test_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='test', seed=model_seed, data_df=data_df)
        if not split_method == 'wt_mutation':
            test_wt_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='test_wt', seed=model_seed, data_df=data_df)
        test_mutation_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='test_mutation', seed=model_seed, data_df=data_df)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(test_set, batch_size=batch_size, shuffle=False)
        if not split_method == 'wt_mutation':
            test_wt_loader = DataLoader(test_wt_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(test_wt_set, batch_size=batch_size, shuffle=False)   
        test_mutation_loader = DataLoader(test_mutation_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(test_mutation_set, batch_size=batch_size, shuffle=False)
        
        model = get_model(model_name=model_name, device=device)
        
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_cindex = AverageMeter()
        running_best_rmse = BestMeter("min")

        early_stop_epoch = 100
        break_flag = False
        
        model.train()

        for epoch in range(epochs):
            if break_flag:
                break

            for data in train_loader:

                if model_name == 'DGraphDTA':
                    data_protien, data_ligand = data[0].to(device), data[1].to(device)
                    pred = model(data_protien, data_ligand)
                    label = data_protien.y
                elif model_name in ['AttentionDTA', 'DeepDTA']:
                    proteins, ligands, label = data
                    proteins, ligands, label = proteins.to(device), ligands.to(device), label.to(device)
                    pred = model(proteins, ligands)
                else:
                    data = data.to(device)
                    pred = model(data)
                    label = data.y

                loss = criterion(pred.view(-1), label.view(-1))
                cindex = get_cindex(label.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.update(loss.item(), label.size(0)) 
                running_cindex.update(cindex, label.size(0))
            
            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            valid_mse, valid_rmse, valid_rp, valid_cindex, _, _ = val(model, valid_loader, device, model_name)

            msg = f"epoch: {epoch}, train_loss: {epoch_loss:.4f}, train_rmse: {epoch_rmse:.4f}, valid_mse: {valid_mse:.4f}, valid_rmse: {valid_rmse:.4f}, valid_rp: {valid_rp:.4f}, valid_cindex: {valid_cindex:.4f}"
            print(msg)

            if valid_rmse < running_best_rmse.get_best():
                running_best_rmse.update(valid_rmse)
                model_path = os.path.join(model_name, 'save', split_method, f'seed_{model_seed}.pt')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
            else:
                count = running_best_rmse.counter()
                if count > early_stop_epoch:
                    best_rmse = running_best_rmse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        load_model_dict(model, model_path, device)
        valid_mse, valid_rmse, valid_rp, valid_cindex, _, _ = val(model, valid_loader, device, model_name)
        test_mse, test_rmse, test_rp, test_cindex, _, _ = val(model, test_loader, device, model_name)
        if not split_method == 'wt_mutation':
            test_wt_mse, test_wt_rmse, test_wt_rp, test_wt_cindex, _, _ = val(model, test_wt_loader, device, model_name)
        test_mutation_mse, test_mutation_rmse, test_mutation_rp, test_mutation_cindex, _, _ = val(model, test_mutation_loader, device, model_name)

        all_test_mse.append(test_mse)
        all_test_rp.append(test_rp)
        all_test_cindex.append(test_cindex)
        if not split_method == 'wt_mutation':
            all_test_wt_mse.append(test_wt_mse)
            all_test_wt_rp.append(test_wt_rp)
            all_test_wt_cindex.append(test_wt_cindex)
        all_test_mutation_mse.append(test_mutation_mse)
        all_test_mutation_rp.append(test_mutation_rp)
        all_test_mutation_cindex.append(test_mutation_cindex)

    print(f"mean test mse: {np.mean(all_test_mse)}")
    print(f"std test mse: {np.std(all_test_mse)}")
    if not split_method == 'wt_mutation':
        print(f"mean test_wt mse: {np.mean(all_test_wt_mse)}")
        print(f"std test_wt mse: {np.std(all_test_wt_mse)}")
    print(f"mean test_mutation mse: {np.mean(all_test_mutation_mse)}")
    print(f"std test_mutation mse: {np.std(all_test_mutation_mse)}")
    print(f"mean test rp: {np.mean(all_test_rp)}")
    print(f"std test rp: {np.std(all_test_rp)}")
    if not split_method == 'wt_mutation':
        print(f"mean test_wt rp: {np.mean(all_test_wt_rp)}")
        print(f"std test_wt rp: {np.std(all_test_wt_rp)}")
    print(f"mean test_mutation rp: {np.mean(all_test_mutation_rp)}")
    print(f"std test_mutation rp: {np.std(all_test_mutation_rp)}")


    
        


        