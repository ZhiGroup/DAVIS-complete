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


def get_model(model_name, model_path, device):
    '''
    model: Literal['MGraphDTA', 'DGraphDTA', 'GraphDTA', 'AttentionDTA', 'DeepDTA']
    '''
    if model_name == 'MGraphDTA':
        from MGraphDTA.model import MGraphDTA
        model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
        load_model_dict(model, model_path, device)
    elif model_name == 'DGraphDTA':
        from DGraphDTA.gnn import GNNNet
        model = GNNNet().to(device)
        load_model_dict(model, model_path, device)
    elif model_name == 'GraphDTA':
        from GraphDTA.models.gat_gcn import GAT_GCN
        model = GAT_GCN().to(device)
        load_model_dict(model, model_path, device)
    elif model_name == 'AttentionDTA':
        from AttentionDTA.model import AttentionDTA
        model = AttentionDTA().to(device)
        load_model_dict(model, model_path, device)
    elif model_name == 'DeepDTA':
        from DeepDTA.model import DeepDTA
        #TODO: add protein and ligand vocab size
        model = DeepDTA(pro_vocab_size=len(train_set.protein_vocab), lig_vocab_size=len(train_set.ligand_vocab), channel=32, protein_kernel_size=12, ligand_kernel_size=8).to(device)
        load_model_dict(model, model_path, device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

def get_dataset(model_name, root, split_method, split, protein, mutation=None, drug=None, drug_type=None, drug_1_type=None, drug_2_type=None, seed=None, data_df=None):
    '''
    model: Literal['MGraphDTA', 'DGraphDTA', 'GraphDTA', 'AttentionDTA', 'DeepDTA']
    '''
    if model_name == 'MGraphDTA':
        from MGraphDTA.preprocessing_fine_tuning import GNNDataset
        dataset = GNNDataset(df=data_df, root=root, split_method=split_method, split=split, protein=protein, mutation=mutation, drug=drug, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed)
    elif model_name == 'DGraphDTA':
        from DGraphDTA.preprocessing_fine_tuning import GNNDataset
        dataset = GNNDataset(df=data_df, root=root, split_method=split_method, split=split, protein=protein, mutation=mutation, drug=drug, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed)
    elif model_name == 'GraphDTA':
        from GraphDTA.preprocessing_fine_tuning import GNNDataset
        dataset = GNNDataset(df=data_df, root=root, split_method=split_method, split=split, protein=protein, mutation=mutation, drug=drug, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed)
    elif model_name == 'AttentionDTA':
        from AttentionDTA.preprocessing_fine_tuning import Dataset
        dataset = Dataset(df=data_df, split_method=split_method, split=split, protein=protein, mutation=mutation, seqlen=1200, smilen=100, drug=drug, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed)
    elif model_name == 'DeepDTA':
        from DeepDTA.preprocessing_fine_tuning import Dataset
        dataset = Dataset(df=data_df, split_method=split_method, split=split, protein=protein, mutation=mutation, drug=drug, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed)
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


def val_wt_groundtruth_baseline(wt_affinity, dataloader, model_name):
    label_list = []
    for data in dataloader:
        if model_name == 'DGraphDTA':
            data_protien, data_ligand = data[0].to(device), data[1].to(device)
            label = data_protien.y
        elif model_name in ['AttentionDTA', 'DeepDTA']:
            proteins, ligands, label = data
            proteins, ligands, label = proteins.to(device), ligands.to(device), label.to(device)
        else: 
            label = data.y
        label_list.append(label.detach().cpu().numpy())
            
    label = np.concatenate(label_list, axis=0)
    if len(label) != len(wt_affinity):
        wt_affinity = np.ones_like(label) * wt_affinity
    else:
        assert len(label) == len(wt_affinity)

    coff = np.corrcoef(wt_affinity, label)[0, 1]
    mse = mean_squared_error(label, wt_affinity)
    rmse = np.sqrt(mean_squared_error(label, wt_affinity))
    cindex = get_cindex(wt_affinity, label)
    
    return mse, rmse, coff, cindex
def get_mutation_name(data_df, protein_name):
    return list(data_df[(data_df['protein'].str.contains(f"{protein_name}_[a-z][0-9]") | data_df['protein'].str.contains(f"{protein_name}_itd") | data_df['protein'].str.contains(f"{protein_name}_p"))]['protein'].unique())

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
    parser.add_argument('--job_name', type=str, default='GIGN_benchmark_davis_complete_ensemble_fine_tuning', help='job name')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model_name', choices=['MGraphDTA', 'DGraphDTA', 'GraphDTA', 'AttentionDTA', 'DeepDTA'], help='model name')
    parser.add_argument('--data_root', type=str, default='data/davis_complete', help='data root')
    parser.add_argument('--data_df', type=str, default='davis_complete.csv', help='data of protein and ligand')
    parser.add_argument('--split_method', choices=['different_mutation_same_drug', 'same_mutation_different_drug', 'different_mutation_different_drug'], help='split method')
    parser.add_argument('--model_seeds', nargs='+', type=int, help='List of seeds for the repeats')
    parser.add_argument('--combination_seed', type=int_or_false, default=0, help='seed for the combination of mutation and drug')
    parser.add_argument('--protein', choices=['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'], help='protein name')
    parser.add_argument('--drug_type', choices=['Type I', 'Type II', 'undetermined'], help='drug type', default=None)
    parser.add_argument('--drug_1_type', choices=['Type I', 'Type II', 'undetermined'], help='drug 1 type for different_mutation_different_drug', default=None)
    parser.add_argument('--drug_2_type', choices=['Type I', 'Type II', 'undetermined'], help='drug 2 type for different_mutation_different_drug', default=None)
    parser.add_argument('--nontruncated_affinity', action='store_true', help='only use nontruncated affinity')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)

    args = parser.parse_args()
    device = torch.device(f'cuda:{str(args.gpu)}')
    model_name = args.model_name
    root = os.path.join(model_name, args.data_root)
    data_df = pd.read_csv(args.data_df)
    nontruncated_affinity = args.nontruncated_affinity
    split_method = args.split_method
    model_seeds = args.model_seeds
    combination_seed = args.combination_seed
    epochs = args.epochs
    lr = args.lr

    # clear the cache 
    if os.path.exists(os.path.join(root, 'processed')):
        for file in os.listdir(os.path.join(root, 'processed')):
            file_path = os.path.join(root, 'processed', file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    all_train_num = []
    all_test_num = []
    all_protein = []
    all_protein_mut_1 = []
    all_protein_mut_2 = []
    

    all_mutation = []
    all_drug_type = []
    all_drug_name = []
    all_drug_1_name = []
    all_drug_1_type = []
    all_drug_2_name = []
    all_drug_2_type = []
    
    all_affinity_wildtype = []
    all_affinity_mut1_drug1 = []
    all_affinity_mut1_drug2 = []
    all_affinity_mut2_drug1 = []
    all_affinity_mut2_drug2 = []

    all_mean_test_mse_wt_groundtruth_baseline = []
    all_std_test_mse_wt_groundtruth_baseline = []
    all_mean_test_mse_wt_prediction_baseline = []
    all_std_test_mse_wt_prediction_baseline = []

    all_mean_test_rp_wt_groundtruth_baseline = []
    all_std_test_rp_wt_groundtruth_baseline = []
    all_mean_test_rp_wt_prediction_baseline = []
    all_std_test_rp_wt_prediction_baseline = []

    all_mean_test_cindex_wt_groundtruth_baseline = []
    all_std_test_cindex_wt_groundtruth_baseline = []
    all_mean_test_cindex_wt_prediction_baseline = []
    all_std_test_cindex_wt_prediction_baseline = []

    all_mean_all_mse_wt_groundtruth_baseline = []
    all_std_all_mse_wt_groundtruth_baseline = []
    all_mean_all_mse_wt_prediction_baseline = []
    all_std_all_mse_wt_prediction_baseline = []

    all_mean_all_rp_wt_groundtruth_baseline = []
    all_std_all_rp_wt_groundtruth_baseline = [] 
    all_mean_all_rp_wt_prediction_baseline = []
    all_std_all_rp_wt_prediction_baseline = []
    
    all_mean_all_cindex_wt_groundtruth_baseline = []
    all_std_all_cindex_wt_groundtruth_baseline = []
    all_mean_all_cindex_wt_prediction_baseline = []
    all_std_all_cindex_wt_prediction_baseline = []

    all_mean_test_mse_ratio_groundtruth_baseline = []
    all_std_test_mse_ratio_groundtruth_baseline = []
    all_mean_test_mse_ratio_prediction_baseline = []
    all_std_test_mse_ratio_prediction_baseline = []

    all_mean_test_mse_original = []
    all_std_test_mse_original = []
    all_mean_all_mse_original = []
    all_std_all_mse_original = []

    all_mean_test_rp_original = []
    all_std_test_rp_original = []
    all_mean_all_rp_original = []
    all_std_all_rp_original = []

    all_mean_test_cindex_original = []
    all_std_test_cindex_original = []
    all_mean_all_cindex_original = []
    all_std_all_cindex_original = []

    all_mean_test_mse_finetuning = []
    all_std_test_mse_finetuning = []

    all_mean_test_rp_finetuning = []
    all_std_test_rp_finetuning = []

    all_mean_test_cindex_finetuning = []
    all_std_test_cindex_finetuning = []
    
    protein = ['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'] 
    ligand = list(pd.read_csv('/data/mwu11/FDA/data/davis_complete/davis_inhibitor_binding_mode.csv')['Compound'])
    drug_type = ['Type I', 'Type II', 'undetermined']
    drug_1_type = ['Type I', 'Type II', 'undetermined']
    drug_2_type = ['Type I', 'Type II', 'undetermined'] 


    if  split_method == 'different_mutation_same_drug' and type(combination_seed) == int:
        combinations = list(product(protein, drug_type))
        combinations.remove(('abl1', 'undetermined'))
        # the maximum number of met is 2, so if train=1, test=1, then we have batch_normailzation error
        combinations.remove(('met', 'Type I'))
        combinations.remove(('met', 'Type II'))
        combinations.remove(('met', 'undetermined'))
        combinations.remove(('pik3ca', 'Type II'))

    elif split_method == 'different_mutation_same_drug' and not combination_seed:
        combinations = list(product(protein, ligand))

    elif split_method == 'same_mutation_different_drug' and not combination_seed:
        combinations = []
        for protein_name in protein:
            for mutation_name in get_mutation_name(data_df, protein_name):
                combinations.append((protein_name, mutation_name))

    elif split_method == 'different_mutation_different_drug' and type(combination_seed) == int:
        combinations = list(product(protein, drug_1_type, drug_2_type))
        combinations.remove(('abl1', 'undetermined', 'undetermined'))
        combinations.remove(('abl1', 'undetermined', 'Type I'))
        combinations.remove(('abl1', 'undetermined', 'Type II'))
        combinations.remove(('abl1', 'Type I', 'undetermined'))
        combinations.remove(('abl1', 'Type II', 'undetermined'))
        combinations.remove(('pik3ca', 'Type I', 'Type II'))
        combinations.remove(('pik3ca', 'Type II', 'Type I'))
        combinations.remove(('pik3ca', 'Type II', 'Type II'))
        combinations.remove(('pik3ca', 'Type II', 'undetermined'))
        combinations.remove(('pik3ca', 'undetermined', 'Type II'))
    else:
        raise ValueError('split_method and combination_seed are not matched')


    for combination in tqdm(combinations):
        print(f'Now we are doing {combination}')
        protein = combination[0]
        if args.split_method == 'different_mutation_same_drug' and type(combination_seed) == int:
            drug_type = combination[1]
            mutation_name = None
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=protein, drug_type=drug_type, df=data_df, seed=combination_seed, nontruncated_affinity=nontruncated_affinity)
            job_name = f'fine_tuning_{split_method}_{protein}_{drug_type}_{drug_name}'
            wt_affinity = data_df[(data_df['protein'] == protein) & (data_df['drug_name'] == drug_name)]['y'].values[0]
            print(f'Now we are doing {job_name}')
        
        elif args.split_method == 'different_mutation_same_drug' and not combination_seed:
            drug_type = None
            mutation_name = None
            drug_name = combination[1]
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=protein, drug=drug_name, df=data_df, seed=combination_seed, nontruncated_affinity=nontruncated_affinity)
            if not split_df:
                continue
            job_name = f'fine_tuning_{split_method}_{protein}_{drug_name}'
            # wt_affinity = data_df[(data_df['protein'] == protein) & (data_df['drug_name'] == drug_name)]['y'].values[0]
            wt_all_affinity = split_df['wt_all']['y'].values
            wt_test_affinity = split_df['wt_test']['y'].values
            print(f'Now we are doing {job_name}')

        elif args.split_method == 'same_mutation_different_drug' and not combination_seed:
            drug_name = None
            mutation_name = combination[1]
            split_df, train_num, test_num = create_fine_tuning_same_mutation_different_drug_split(protein=protein, mutation=mutation_name, df=data_df, nontruncated_affinity=nontruncated_affinity)
            if not split_df:
                continue
            job_name = f'fine_tuning_{split_method}_{protein}_{mutation_name}'
            wt_all_affinity = split_df['wt_all']['y'].values
            wt_test_affinity = split_df['wt_test']['y'].values
            print(f'Now we are doing {job_name}')

        elif args.split_method == 'different_mutation_different_drug':
            drug_name = None
            mutation_name = None
            drug_1_type = combination[1]
            drug_2_type = combination[2]
            split_df, mut_1, mut_2, drug_1_name, drug_2_name = create_fine_tuning_different_mutation_different_drug_split(protein=protein, drug_1_type=drug_1_type, drug_2_type=drug_2_type, df=data_df, seed=combination_seed, nontruncated_affinity=nontruncated_affinity)
            job_name = f'fine_tuning_{split_method}_{protein}_{mut_1}_{mut_2}_{drug_1_type}_{drug_1_name}_{drug_2_type}_{drug_2_name}'
            train_df, test_df, wt_df = split_df['train'], split_df['test'], split_df['wt']
            wt_affinity = data_df[(data_df['protein'] == protein) & (data_df['drug_name'] == drug_2_name)]['y'].values[0]

            affinity_mut1_drug1 = train_df[(train_df['protein'] == mut_1) & (train_df['drug_name'] == drug_1_name)]['y'].values[0]
            affinity_mut1_drug2 = train_df[(train_df['protein'] == mut_1) & (train_df['drug_name'] == drug_2_name)]['y'].values[0]
            affinity_mut2_drug1 = train_df[(train_df['protein'] == mut_2) & (train_df['drug_name'] == drug_1_name)]['y'].values[0]
            affinity_mut2_drug2 = test_df[(test_df['protein'] == mut_2) & (test_df['drug_name'] == drug_2_name)]['y'].values[0]

            all_affinity_mut1_drug1.append(affinity_mut1_drug1)
            all_affinity_mut1_drug2.append(affinity_mut1_drug2)
            all_affinity_mut2_drug1.append(affinity_mut2_drug1)
            all_affinity_mut2_drug2.append(affinity_mut2_drug2)

            print(f'Now we are doing {job_name}')
        else:
            raise ValueError('split_method and combination_seed are not matched')
        

        print(f'Now we are doing {combination}')

        for model_seed in model_seeds:

            if os.path.exists(os.path.join(model_name, 'save', split_method, job_name, f'seed_{model_seed}.pt')):
                print(f"Model {model_seed} already exists, skip training")
                continue

            train_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='train', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, data_df=data_df)
            train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(train_set, batch_size=len(train_set), shuffle=True)
            model_path = os.path.join(model_name, 'save', 'wt_mutation', f'seed_{model_seed}.pt')
            model = get_model(model_name=model_name, model_path=model_path, device=device)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            running_loss = AverageMeter()
            running_cindex = AverageMeter()
            running_best_mse = BestMeter("min")  

            model.train()

            for epoch in range(epochs):
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
 
            # save the last model
            model_path = os.path.join(model_name, 'save', split_method, job_name, f'seed_{model_seed}.pt')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    
        all_test_mse_wt_groundtruth_baseline = []
        all_test_rp_wt_groundtruth_baseline = []
        all_test_cindex_wt_groundtruth_baseline = []

        all_all_mse_wt_groundtruth_baseline = []
        all_all_rp_wt_groundtruth_baseline = []
        all_all_cindex_wt_groundtruth_baseline = []

        all_test_mse_ratio_groundtruth_baseline = []
        all_test_rmse_ratio_groundtruth_baseline = []

        all_test_mse_wt_prediction_baseline = []
        all_test_rp_wt_prediction_baseline = []
        all_test_cindex_wt_prediction_baseline = []

        all_all_mse_wt_prediction_baseline = []
        all_all_rp_wt_prediction_baseline = []
        all_all_cindex_wt_prediction_baseline = []

        all_test_mse_ratio_prediction_baseline = []
        all_test_rmse_ratio_prediction_baseline = []

        all_test_mse_original = []
        all_test_rp_original = []
        all_test_cindex_original = []
        all_all_mse_original = []
        all_all_rp_original = []
        all_all_cindex_original = []

        all_test_mse_finetuning = []
        all_test_rp_finetuning = []
        all_test_cindex_finetuning = []


        for model_seed in model_seeds:
            all_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='all', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, data_df=data_df)
            train_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='train', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, data_df=data_df)
            test_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='test', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, data_df=data_df)
            wt_all_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='wt_all', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, data_df=data_df)
            wt_test_set = get_dataset(model_name=model_name, root=root, split_method=split_method, split='wt_test', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, data_df=data_df)
            
            all_loader = DataLoader(all_set, batch_size=len(all_set), shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(all_set, batch_size=len(all_set), shuffle=False)
            train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(train_set, batch_size=len(train_set), shuffle=False)
            test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(test_set, batch_size=len(test_set), shuffle=False)
            wt_all_loader = DataLoader(wt_all_set, batch_size=len(wt_all_set), shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(wt_all_set, batch_size=len(wt_all_set), shuffle=False)
            wt_test_loader = DataLoader(wt_test_set, batch_size=len(wt_test_set), shuffle=False, collate_fn=collate_fn) if model_name in ['AttentionDTA', 'DeepDTA'] else DataLoader(wt_test_set, batch_size=len(wt_test_set), shuffle=False)

            model_path_original = os.path.join(model_name, 'save', 'wt_mutation', f'seed_{model_seed}.pt')
            model_path_finetuning = os.path.join(model_name, 'save', split_method, job_name, f'seed_{model_seed}.pt')
            model_original = get_model(model_name=model_name, model_path=model_path_original, device=device)
            model_finetuning = get_model(model_name=model_name, model_path=model_path_finetuning, device=device)
            
            test_mse_wt_groundtruth_baseline, test_rmse_wt_groundtruth_baseline, test_rp_wt_groundtruth_baseline, test_cindex_wt_groundtruth_baseline = val_wt_groundtruth_baseline(wt_test_affinity, test_loader, model_name)
            _, _, _, _, wt_test_prediction, _ = val(model_original, wt_test_loader, device, model_name)
            test_mse_wt_prediction_baseline, test_rmse_wt_prediction_baseline, test_rp_wt_prediction_baseline, test_cindex_wt_prediction_baseline = val_wt_groundtruth_baseline(wt_test_prediction, test_loader, model_name)
            
            all_mse_wt_groundtruth_baseline, all_rmse_wt_groundtruth_baseline, all_rp_wt_groundtruth_baseline, all_cindex_wt_groundtruth_baseline = val_wt_groundtruth_baseline(wt_all_affinity, all_loader, model_name)
            _, _, _, _, wt_all_prediction, _ = val(model_original, wt_all_loader, device, model_name)
            all_mse_wt_prediction_baseline, all_rmse_wt_prediction_baseline, all_rp_wt_prediction_baseline, all_cindex_wt_prediction_baseline = val_wt_groundtruth_baseline(wt_all_prediction, all_loader, model_name)
            
            if args.split_method == 'different_mutation_different_drug':
                ratio_groundtruth = (affinity_mut1_drug2 * affinity_mut2_drug1) / affinity_mut1_drug1
                test_mse_ratio_groundtruth_baseline, test_rmse_ratio_groundtruth_baseline, test_rp_ratio_groundtruth_baseline = val_wt_groundtruth_baseline(ratio_groundtruth, test_loader, model_name)
                _, _, _, _, mut_lig_prediction, _ = val(model_original, train_loader, device, model_name)
                ratio_prediction = (mut_lig_prediction[1] * mut_lig_prediction[2]) / mut_lig_prediction[0]
                test_mse_ratio_prediction_baseline, test_rmse_ratio_prediction_baseline, test_rp_ratio_prediction_baseline = val_wt_groundtruth_baseline(ratio_prediction, test_loader, model_name)
            
            test_mse_original, test_rmse_original, test_rp_original, test_cindex_original, prediction_original, label = val(model_original, test_loader, device, model_name)
            test_mse_finetuning, test_rmse_finetuning, test_rp_finetuning, test_cindex_finetuning, prediction_finetuning, label = val(model_finetuning, test_loader, device, model_name)
            all_mse_original, all_rmse_original, all_rp_original, all_cindex_original, prediction_original_all, label_all = val(model_original, all_loader, device, model_name)
            

            if args.split_method == 'different_mutation_same_drug':
                print(f'label: {label}')
                print(f'gt_wt: {wt_all_affinity}')
                print(f'prediction_wt: {wt_all_prediction}')
                print(f'prediction_original: {prediction_original}')
                print(f'prediction_finetuning: {prediction_finetuning}')

            elif args.split_method == 'same_mutation_different_drug':
                print(f'label: {label}')
                print(f'gt_wt: {wt_test_affinity}')
                print(f'prediction_wt: {wt_test_prediction}')
                print(f'prediction_original: {prediction_original}')
                print(f'prediction_finetuning: {prediction_finetuning}')

            elif args.split_method == 'different_mutation_different_drug':
                print(f'label: {label}')
                print(f'gt_wt: {wt_affinity}')
                print(f'prediction_wt: {wt_prediction}')
                print(f'gt_ratio: {ratio_groundtruth}')
                print(f'prediction_ratio: {ratio_prediction}')
                print(f'prediction_original: {prediction_original}')
                print(f'prediction_finetuning: {prediction_finetuning}')

            msg = f"model_seed: {model_seed}, test_mse_original: {test_mse_original:.4f}, test_rmse_original: {test_rmse_original:.4f}, test_rp_original: {test_rp_original:.4f}, test_cindex_original: {test_cindex_original:.4f},\
                    test_mse_finetuning: {test_mse_finetuning:.4f}, test_rmse_finetuning: {test_rmse_finetuning:.4f}, test_rp_finetuning: {test_rp_finetuning:.4f}, test_cindex_finetuning: {test_cindex_finetuning:.4f}"
            print(msg)
            
            all_test_mse_wt_groundtruth_baseline.append(test_mse_wt_groundtruth_baseline)
            all_test_rp_wt_groundtruth_baseline.append(test_rp_wt_groundtruth_baseline)
            all_test_cindex_wt_groundtruth_baseline.append(test_cindex_wt_groundtruth_baseline)

            all_test_mse_wt_prediction_baseline.append(test_mse_wt_prediction_baseline)
            all_test_rp_wt_prediction_baseline.append(test_rp_wt_prediction_baseline)
            all_test_cindex_wt_prediction_baseline.append(test_cindex_wt_prediction_baseline)
            
            all_all_mse_wt_groundtruth_baseline.append(all_mse_wt_groundtruth_baseline)
            all_all_rp_wt_groundtruth_baseline.append(all_rp_wt_groundtruth_baseline)
            all_all_cindex_wt_groundtruth_baseline.append(all_cindex_wt_groundtruth_baseline)
            
            all_all_mse_wt_prediction_baseline.append(all_mse_wt_prediction_baseline)
            all_all_rp_wt_prediction_baseline.append(all_rp_wt_prediction_baseline)
            all_all_cindex_wt_prediction_baseline.append(all_cindex_wt_prediction_baseline)

            if args.split_method == 'different_mutation_different_drug':
                all_test_mse_ratio_groundtruth_baseline.append(test_mse_ratio_groundtruth_baseline)
                all_test_mse_ratio_prediction_baseline.append(test_mse_ratio_prediction_baseline)
                
            all_test_mse_original.append(test_mse_original)
            all_test_rp_original.append(test_rp_original)
            all_test_cindex_original.append(test_cindex_original)
            all_all_mse_original.append(all_mse_original)
            all_all_rp_original.append(all_rp_original)
            all_all_cindex_original.append(all_cindex_original)

            all_test_mse_finetuning.append(test_mse_finetuning)
            all_test_rp_finetuning.append(test_rp_finetuning)
            all_test_cindex_finetuning.append(test_cindex_finetuning)
        
        
        
        all_protein.append(protein)

        if args.split_method == 'different_mutation_same_drug':
            all_drug_type.append(drug_type)
            all_drug_name.append(drug_name)
            all_train_num.append(train_num)
            all_test_num.append(test_num)
        elif args.split_method == 'same_mutation_different_drug':
            all_mutation.append(mutation_name)
            all_train_num.append(train_num)
            all_test_num.append(test_num)
        else:
            all_protein_mut_1.append(mut_1)
            all_protein_mut_2.append(mut_2)
            all_drug_1_name.append(drug_1_name)
            all_drug_1_type.append(drug_1_type)
            all_drug_2_name.append(drug_2_name)
            all_drug_2_type.append(drug_2_type)

        all_mean_test_mse_wt_groundtruth_baseline.append(np.mean(all_test_mse_wt_groundtruth_baseline))
        all_std_test_mse_wt_groundtruth_baseline.append(np.std(all_test_mse_wt_groundtruth_baseline))
        all_mean_test_mse_wt_prediction_baseline.append(np.mean(all_test_mse_wt_prediction_baseline))
        all_std_test_mse_wt_prediction_baseline.append(np.std(all_test_mse_wt_prediction_baseline))

        all_mean_test_rp_wt_groundtruth_baseline.append(np.mean(all_test_rp_wt_groundtruth_baseline))
        all_std_test_rp_wt_groundtruth_baseline.append(np.std(all_test_rp_wt_groundtruth_baseline))
        all_mean_test_rp_wt_prediction_baseline.append(np.mean(all_test_rp_wt_prediction_baseline))
        all_std_test_rp_wt_prediction_baseline.append(np.std(all_test_rp_wt_prediction_baseline))

        all_mean_test_cindex_wt_groundtruth_baseline.append(np.mean(all_test_cindex_wt_groundtruth_baseline))
        all_std_test_cindex_wt_groundtruth_baseline.append(np.std(all_test_cindex_wt_groundtruth_baseline))
        all_mean_test_cindex_wt_prediction_baseline.append(np.mean(all_test_cindex_wt_prediction_baseline))
        all_std_test_cindex_wt_prediction_baseline.append(np.std(all_test_cindex_wt_prediction_baseline))
        
        all_mean_all_mse_wt_groundtruth_baseline.append(np.mean(all_all_mse_wt_groundtruth_baseline))
        all_std_all_mse_wt_groundtruth_baseline.append(np.std(all_all_mse_wt_groundtruth_baseline))
        all_mean_all_mse_wt_prediction_baseline.append(np.mean(all_all_mse_wt_prediction_baseline))
        all_std_all_mse_wt_prediction_baseline.append(np.std(all_all_mse_wt_prediction_baseline))

        all_mean_all_rp_wt_groundtruth_baseline.append(np.mean(all_all_rp_wt_groundtruth_baseline))
        all_std_all_rp_wt_groundtruth_baseline.append(np.std(all_all_rp_wt_groundtruth_baseline))
        all_mean_all_rp_wt_prediction_baseline.append(np.mean(all_all_rp_wt_prediction_baseline))
        all_std_all_rp_wt_prediction_baseline.append(np.std(all_all_rp_wt_prediction_baseline))

        all_mean_all_cindex_wt_groundtruth_baseline.append(np.mean(all_all_cindex_wt_groundtruth_baseline))
        all_std_all_cindex_wt_groundtruth_baseline.append(np.std(all_all_cindex_wt_groundtruth_baseline))
        all_mean_all_cindex_wt_prediction_baseline.append(np.mean(all_all_cindex_wt_prediction_baseline))
        all_std_all_cindex_wt_prediction_baseline.append(np.std(all_all_cindex_wt_prediction_baseline))

        if args.split_method == 'different_mutation_different_drug':
            all_mean_test_mse_ratio_groundtruth_baseline.append(np.mean(all_test_mse_ratio_groundtruth_baseline))
            all_std_test_mse_ratio_groundtruth_baseline.append(np.std(all_test_mse_ratio_groundtruth_baseline))
            all_mean_test_mse_ratio_prediction_baseline.append(np.mean(all_test_mse_ratio_prediction_baseline))
            all_std_test_mse_ratio_prediction_baseline.append(np.std(all_test_mse_ratio_prediction_baseline))
        
        all_mean_test_mse_original.append(np.mean(all_test_mse_original))
        all_std_test_mse_original.append(np.std(all_test_mse_original))
        all_mean_test_rp_original.append(np.mean(all_test_rp_original))
        all_std_test_rp_original.append(np.std(all_test_rp_original))
        all_mean_test_cindex_original.append(np.mean(all_test_cindex_original))
        all_std_test_cindex_original.append(np.std(all_test_cindex_original))

        all_mean_all_mse_original.append(np.mean(all_all_mse_original))
        all_std_all_mse_original.append(np.std(all_all_mse_original))
        all_mean_all_rp_original.append(np.mean(all_all_rp_original))
        all_std_all_rp_original.append(np.std(all_all_rp_original))
        all_mean_all_cindex_original.append(np.mean(all_all_cindex_original))
        all_std_all_cindex_original.append(np.std(all_all_cindex_original))

        all_mean_test_mse_finetuning.append(np.mean(all_test_mse_finetuning))
        all_std_test_mse_finetuning.append(np.std(all_test_mse_finetuning))
        all_mean_test_rp_finetuning.append(np.mean(all_test_rp_finetuning))
        all_std_test_rp_finetuning.append(np.std(all_test_rp_finetuning))
        all_mean_test_cindex_finetuning.append(np.mean(all_test_cindex_finetuning))
        all_std_test_cindex_finetuning.append(np.std(all_test_cindex_finetuning))

    
    if args.split_method == 'different_mutation_same_drug':
        df = pd.DataFrame({'protein': all_protein, 
                        'drug_type': all_drug_type, 
                        'drug_name': all_drug_name, 
                        'train_num': all_train_num, 
                        'test_num': all_test_num, 
                        'mean_test_mse_wt_groundtruth_baseline': all_mean_test_mse_wt_groundtruth_baseline, 
                        'std_test_mse_wt_groundtruth_baseline': all_std_test_mse_wt_groundtruth_baseline, 
                        'mean_test_mse_wt_prediction_baseline': all_mean_test_mse_wt_prediction_baseline, 
                        'std_test_mse_wt_prediction_baseline': all_std_test_mse_wt_prediction_baseline, 
                        'mean_test_mse_original': all_mean_test_mse_original, 
                        'std_test_mse_original': all_std_test_mse_original, 
                        'mean_test_mse_finetuning': all_mean_test_mse_finetuning, 
                        'std_test_mse_finetuning': all_std_test_mse_finetuning, 
                        'mean_test_rp_original': all_mean_test_rp_original, 
                        'std_test_rp_original': all_std_test_rp_original, 
                        'mean_test_rp_finetuning': all_mean_test_rp_finetuning, 
                        'std_test_rp_finetuning': all_std_test_rp_finetuning, 
                        'mean_test_cindex_original': all_mean_test_cindex_original, 
                        'std_test_cindex_original': all_std_test_cindex_original, 
                        'mean_test_cindex_finetuning': all_mean_test_cindex_finetuning, 
                        'std_test_cindex_finetuning': all_std_test_cindex_finetuning, 
                        'mean_all_mse_wt_groundtruth_baseline': all_mean_all_mse_wt_groundtruth_baseline, 
                        'std_all_mse_wt_groundtruth_baseline': all_std_all_mse_wt_groundtruth_baseline, 
                        'mean_all_mse_wt_prediction_baseline': all_mean_all_mse_wt_prediction_baseline, 
                        'std_all_mse_wt_prediction_baseline': all_std_all_mse_wt_prediction_baseline, 
                        'mean_all_mse_original': all_mean_all_mse_original, 
                        'std_all_mse_original': all_std_all_mse_original,
                        'mean_all_rp_original': all_mean_all_rp_original,
                        'std_all_rp_original': all_std_all_rp_original,
                        'mean_all_cindex_original': all_mean_all_cindex_original,
                        'std_all_cindex_original': all_std_all_cindex_original})
    elif args.split_method == 'same_mutation_different_drug':
        df = pd.DataFrame({'protein': all_protein,
                           'mutation': all_mutation,
                           'train_num': all_train_num,
                           'test_num': all_test_num,
                           'mean_test_mse_wt_groundtruth_baseline': all_mean_test_mse_wt_groundtruth_baseline,
                           'std_test_mse_wt_groundtruth_baseline': all_std_test_mse_wt_groundtruth_baseline,
                           'mean_test_mse_wt_prediction_baseline': all_mean_test_mse_wt_prediction_baseline,
                           'std_test_mse_wt_prediction_baseline': all_std_test_mse_wt_prediction_baseline,
                           'mean_test_rp_wt_groundtruth_baseline': all_mean_test_rp_wt_groundtruth_baseline,
                           'std_test_rp_wt_groundtruth_baseline': all_std_test_rp_wt_groundtruth_baseline,
                           'mean_test_rp_wt_prediction_baseline': all_mean_test_rp_wt_prediction_baseline,
                           'std_test_rp_wt_prediction_baseline': all_std_test_rp_wt_prediction_baseline,
                           'mean_test_cindex_wt_groundtruth_baseline': all_mean_test_cindex_wt_groundtruth_baseline,
                           'std_test_cindex_wt_groundtruth_baseline': all_std_test_cindex_wt_groundtruth_baseline,
                           'mean_test_cindex_wt_prediction_baseline': all_mean_test_cindex_wt_prediction_baseline,
                           'std_test_cindex_wt_prediction_baseline': all_std_test_cindex_wt_prediction_baseline,
                           'mean_test_mse_original': all_mean_test_mse_original,
                           'std_test_mse_original': all_std_test_mse_original,
                           'mean_test_mse_finetuning': all_mean_test_mse_finetuning,
                           'std_test_mse_finetuning': all_std_test_mse_finetuning,
                           'mean_test_rp_original': all_mean_test_rp_original,
                           'std_test_rp_original': all_std_test_rp_original,
                           'mean_test_rp_finetuning': all_mean_test_rp_finetuning,
                           'std_test_rp_finetuning': all_std_test_rp_finetuning,
                           'mean_test_cindex_original': all_mean_test_cindex_original,
                           'std_test_cindex_original': all_std_test_cindex_original,
                           'mean_test_cindex_finetuning': all_mean_test_cindex_finetuning,
                           'std_test_cindex_finetuning': all_std_test_cindex_finetuning,
                           'mean_all_mse_wt_groundtruth_baseline': all_mean_all_mse_wt_groundtruth_baseline,
                           'std_all_mse_wt_groundtruth_baseline': all_std_all_mse_wt_groundtruth_baseline,
                           'mean_all_mse_wt_prediction_baseline': all_mean_all_mse_wt_prediction_baseline,
                           'std_all_mse_wt_prediction_baseline': all_std_all_mse_wt_prediction_baseline,
                           'mean_all_rp_wt_groundtruth_baseline': all_mean_all_rp_wt_groundtruth_baseline,
                           'std_all_rp_wt_groundtruth_baseline': all_std_all_rp_wt_groundtruth_baseline,
                           'mean_all_rp_wt_prediction_baseline': all_mean_all_rp_wt_prediction_baseline,
                           'std_all_rp_wt_prediction_baseline': all_std_all_rp_wt_prediction_baseline,
                           'mean_all_cindex_wt_groundtruth_baseline': all_mean_all_cindex_wt_groundtruth_baseline,
                           'std_all_cindex_wt_groundtruth_baseline': all_std_all_cindex_wt_groundtruth_baseline,
                           'mean_all_cindex_wt_prediction_baseline': all_mean_all_cindex_wt_prediction_baseline,
                           'std_all_cindex_wt_prediction_baseline': all_std_all_cindex_wt_prediction_baseline,
                           'mean_all_mse_original': all_mean_all_mse_original,
                           'std_all_mse_original': all_std_all_mse_original,
                           'mean_all_rp_original': all_mean_all_rp_original,
                           'std_all_rp_original': all_std_all_rp_original,
                           'mean_all_cindex_original': all_mean_all_cindex_original,
                           'std_all_cindex_original': all_std_all_cindex_original})
    elif args.split_method == 'different_mutation_different_drug':
        df = pd.DataFrame({'protein': all_protein, 
                            'mut_1': all_protein_mut_1, 
                            'mut_2': all_protein_mut_2, 
                            'drug_1_type': all_drug_1_type, 
                            'drug_1_name': all_drug_1_name, 
                            'drug_2_type': all_drug_2_type, 
                            'drug_2_name': all_drug_2_name, 
                            'affinity_mut1_drug1': all_affinity_mut1_drug1, 
                            'affinity_mut1_drug2': all_affinity_mut1_drug2, 
                            'affinity_mut2_drug1': all_affinity_mut2_drug1, 
                            'affinity_mut2_drug2': all_affinity_mut2_drug2, 
                            'mean_test_mse_wt_groundtruth_baseline': all_mean_test_mse_wt_groundtruth_baseline, 
                            'std_test_mse_wt_groundtruth_baseline': all_std_test_mse_wt_groundtruth_baseline, 
                            'mean_test_mse_wt_prediction_baseline': all_mean_test_mse_wt_prediction_baseline, 
                            'std_test_mse_wt_prediction_baseline': all_std_test_mse_wt_prediction_baseline, 
                            'mean_test_mse_ratio_groundtruth_baseline': all_mean_test_mse_ratio_groundtruth_baseline, 
                            'std_test_mse_ratio_groundtruth_baseline': all_std_test_mse_ratio_groundtruth_baseline, 
                            'mean_test_mse_ratio_prediction_baseline': all_mean_test_mse_ratio_prediction_baseline, 
                            'std_test_mse_ratio_prediction_baseline': all_std_test_mse_ratio_prediction_baseline, 
                            'mean_test_mse_original': all_mean_test_mse_original, 
                            'std_test_mse_original': all_std_test_mse_original, 
                            'mean_test_mse_finetuning': all_mean_test_mse_finetuning, 
                            'std_test_mse_finetuning': all_std_test_mse_finetuning, 
                            'mean_test_rp_original': all_mean_test_rp_original, 
                            'std_test_rp_original': all_std_test_rp_original, 
                            'mean_test_rp_finetuning': all_mean_test_rp_finetuning, 
                            'std_test_rp_finetuning': all_std_test_rp_finetuning, 
                            'mean_test_cindex_original': all_mean_test_cindex_original,
                            'std_test_cindex_original': all_std_test_cindex_original, 
                            'mean_test_cindex_finetuning': all_mean_test_cindex_finetuning, 
                            'std_test_cindex_finetuning': all_std_test_cindex_finetuning})
    else:
        raise ValueError('split_method is not matched')
   
    df.to_csv(f'{model_name}_result_finetuning_{split_method}_epoch{epochs}_lr{lr}_combinationseed{combination_seed}.csv', index=False)