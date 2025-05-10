
# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from dataset_GIGN_benchmark_davis_complete_fine_tuning import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from argparse import Namespace, ArgumentParser, FileType, ArgumentTypeError
from joblib import Parallel, delayed, parallel_backend
from joblib.externals.loky.backend.context import get_context
import warnings
import pickle
from itertools import product
from tqdm import tqdm
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
from scipy.stats import kendalltau

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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



# %%
def val_ensemble(models, dataloader, device):
    for model in models:
        model.eval()
    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        label = data.y
        ensemble_pred = []
        for i in range(args_.ensemble_size):
            with torch.no_grad():
                model = models[i]
                pred = model(data)
                ensemble_pred.append(pred.detach().cpu().numpy())

        pred_list.append(np.mean(ensemble_pred, axis=0))
        label_list.append(label.detach().cpu().numpy())

    if not pred_list:
        return 0, 0, 0

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    cindex = get_cindex(pred, label)
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    

    return mse, rmse, coff, cindex, pred, label

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    cindex = get_cindex(pred, label)
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    

    model.train()

    return mse, rmse, coff, cindex, pred, label


def val_wt_groundtruth_baseline(wt_affinity, dataloader):
    label_list = []
    for data in dataloader:
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


def train_one_model(job_name, model_idx, model, optimizer, criterion, running_loss, model_list,
                    seed, train_loader, test_loader, device, epochs, logger, save_model):
    """
    Train a single model for 'epochs' epochs and return the best RMSE or any stats you need.
    """
    break_flag = False
    for epoch in range(epochs):
        # Training loop
        model.train()
        for data in train_loader:
            
            data = data.to(device)
            pred = model(data)
            label = data.y

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.update(loss.item(), label.size(0))

        epoch_loss = running_loss.get_average()
        epoch_rmse = np.sqrt(epoch_loss)
        running_loss.reset()

        # Validation
        test_mse, test_rmse, test_rp, test_cindex, _, _= val(model, test_loader, device)
        # msg = (
        #     f"train_rmse-{epoch_rmse:.4f}, test_mse-{test_mse:.4f}, test_rmse-{test_rmse:.4f}, test_rp-{test_rp:.4f}"
        # )
        # print(msg)

        # Save best model

        if save_model:
            msg_save = f"model_{model_idx}_epoch_{epoch}"
            model_path = os.path.join('model', job_name, f'seed_{seed}', msg_save + '.pt')
            model_list.append(model_path)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

    return {
        "model_idx": model_idx,
        "model_paths": model_list
    }




# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN_benchmark'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    # repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    # early_stop_epoch = 10
    logger = TrainLogger(args, cfg, create=True)
    
    parser = ArgumentParser()
    parser.add_argument('--job_name', type=str, default='GIGN_benchmark_davis_complete_ensemble_fine_tuning', help='job name')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--data_df', type=str, default='../../data/davis_complete/davis_complete.tsv', help='data of protein and ligand')
    parser.add_argument('--complex_path', type=str, default='../../data/davis_complete/alphafold_structure_kinase_domain', help='the path of the complexes')
    parser.add_argument('--split_method', choices=['different_mutation_same_drug', 'same_mutation_different_drug', 'different_mutation_different_drug'], help='split method')
    parser.add_argument('--model_seeds', nargs='+', type=int, help='List of seeds for the repeats')
    parser.add_argument('--combination_seed', type=int_or_false, default=False, help='seed for the combination of mutation and drug')
    parser.add_argument('--protein', choices=['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'], help='protein name')
    parser.add_argument('--drug_type', choices=['Type I', 'Type II', 'undetermined'], help='drug type', default=None)
    parser.add_argument('--drug_1_type', choices=['Type I', 'Type II', 'undetermined'], help='drug 1 type for different_mutation_different_drug', default=None)
    parser.add_argument('--drug_2_type', choices=['Type I', 'Type II', 'undetermined'], help='drug 2 type for different_mutation_different_drug', default=None)
    parser.add_argument('--nontruncated_affinity', action='store_true', default=True, help='only use nontruncated affinity')
    parser.add_argument('--ensemble_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    
    
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


    args_ = parser.parse_args()
    device = torch.device(f'cuda:{str(args_.gpu)}')
    data_root = args_.complex_path
    data_df = pd.read_csv(args_.data_df , sep='\t')
    nontruncated_affinity = args_.nontruncated_affinity
    split_method = args_.split_method
    model_seeds = args_.model_seeds
    combination_seed = args_.combination_seed
    epochs = args_.epochs
    lr = args_.lr
    
    # protein = ['abl1', 'egfr', 'flt3', 'kit', 'met', 'pik3ca', 'ret']
    ## remove gcn2, as it does not have corresponding wild-type
    protein = ['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'] 
    ligand = list(pd.read_csv('../../data/davis_complete/davis_inhibitor_binding_mode.csv')['Compound'])
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
        if args_.split_method == 'different_mutation_same_drug' and type(combination_seed) == int:
            drug_type = combination[1]
            mutation_name = None
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=protein, drug_type=drug_type, df=data_df, seed=combination_seed, nontruncated_affinity=nontruncated_affinity)
            job_name = f'fine_tuning_{split_method}_{protein}_{drug_type}_{drug_name}'
            wt_affinity = data_df[(data_df['protein'] == protein) & (data_df['drug_name'] == drug_name)]['y'].values[0]
            print(f'Now we are doing {job_name}')
        
        elif args_.split_method == 'different_mutation_same_drug' and not combination_seed:
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

        elif args_.split_method == 'same_mutation_different_drug' and not combination_seed:
            drug_name = None
            mutation_name = combination[1]
            split_df, train_num, test_num = create_fine_tuning_same_mutation_different_drug_split(protein=protein, mutation=mutation_name, df=data_df, nontruncated_affinity=nontruncated_affinity)
            if not split_df:
                continue
            job_name = f'fine_tuning_{split_method}_{protein}_{mutation_name}'
            wt_all_affinity = split_df['wt_all']['y'].values
            wt_test_affinity = split_df['wt_test']['y'].values
            print(f'Now we are doing {job_name}')

        elif args_.split_method == 'different_mutation_different_drug':
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

        for model_seed in model_seeds:
            if os.path.exists(f'model/{job_name}/results_seed_{model_seed}.pkl'):
                print(f'model/{job_name}/results_seed_{model_seed}.pkl already exists, skip this seed')
                continue
            
            models = [GIGN(35, 256).to(device) for _ in range(args_.ensemble_size)]
            results = read_pickle(f'model/benchmark_wt_mutation_davis_complete_ensemble_testsplit/results_seed_{model_seed}.pkl')
            for res in results:
                model = models[res['model_idx']]
                load_model_dict(model, res['model_paths'][-1])
            train_set = GraphDataset(data_root, data_df, split_method=split_method, split='train', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
            test_set = GraphDataset(data_root, data_df, split_method=split_method, split='test', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
        
            train_loader = PLIDataLoader(train_set, batch_size=len(train_set), shuffle=True)
            test_loader = PLIDataLoader(test_set, batch_size=len(test_set), shuffle=False)
            

            logger.info(f"this is the model seed {model_seed}")
            logger.info(__file__)
            logger.info(f"split method: {args_.split_method}")
            logger.info(f"train data: {len(train_set)}")
            logger.info(f"test data: {len(test_set)}")
            
            optimizers = [optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6) for model in models]
            criterions = [nn.MSELoss() for _ in range(args_.ensemble_size)]

            running_losses = [AverageMeter() for _ in range(args_.ensemble_size)]
            model_lists = [[] for _ in range(args_.ensemble_size)]
            
            pool_input = []
            for i in range(args_.ensemble_size):
                pool_input.append((job_name, i, models[i], 
                                    optimizers[i], 
                                    criterions[i], 
                                    running_losses[i], 
                                    model_lists[i],
                                    model_seed, 
                                    train_loader, 
                                    test_loader, 
                                    device, 
                                    epochs, 
                                    logger, 
                                    save_model))
            with parallel_backend('loky', n_jobs=args_.ensemble_size):
                results = Parallel()(delayed(train_one_model)(*input_) for input_ in pool_input)


            with open(f'model/{job_name}/results_seed_{model_seed}.pkl', 'wb') as f:
                pickle.dump(results, f)

        
            # for res in results:
            #     model = models[res['model_idx']]
            #     load_model_dict(model, res['model_paths'][-1])

        
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
            all_set = GraphDataset(data_root, data_df, split_method=split_method, split='all', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
            train_set = GraphDataset(data_root, data_df, split_method=split_method, split='train', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
            test_set = GraphDataset(data_root, data_df, split_method=split_method, split='test', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
            wt_all_set = GraphDataset(data_root, data_df, split_method=split_method, split='wt_all', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
            wt_test_set = GraphDataset(data_root, data_df, split_method=split_method, split='wt_test', protein=protein, mutation=mutation_name, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed, create=False)
        
            all_loader = PLIDataLoader(all_set, batch_size=len(all_set), shuffle=False)
            train_loader = PLIDataLoader(train_set, batch_size=len(train_set), shuffle=False)
            test_loader = PLIDataLoader(test_set, batch_size=len(test_set), shuffle=False)
            wt_all_loader = PLIDataLoader(wt_all_set, batch_size=len(wt_all_set), shuffle=False)
            wt_test_loader = PLIDataLoader(wt_test_set, batch_size=len(wt_test_set), shuffle=False)


            models_original = [GIGN(35, 256).to(device) for _ in range(args_.ensemble_size)]
            results = read_pickle(f'model/benchmark_wt_mutation_davis_complete_ensemble_testsplit/results_seed_{model_seed}.pkl')
            
            for res in results:
                model = models_original[res['model_idx']]
                load_model_dict(model, res['model_paths'][-1])

            models_finetuning = [GIGN(35, 256).to(device) for _ in range(args_.ensemble_size)]
            results = read_pickle(f'model/{job_name}/results_seed_{model_seed}.pkl')
            
            for res in results:
                model = models_finetuning[res['model_idx']]
                load_model_dict(model, res['model_paths'][-1])
            
            test_mse_wt_groundtruth_baseline, test_rmse_wt_groundtruth_baseline, test_rp_wt_groundtruth_baseline, test_cindex_wt_groundtruth_baseline = val_wt_groundtruth_baseline(wt_test_affinity, test_loader)
            _, _, _, _, wt_test_prediction, _ = val_ensemble(models_original, wt_test_loader, device)
            test_mse_wt_prediction_baseline, test_rmse_wt_prediction_baseline, test_rp_wt_prediction_baseline, test_cindex_wt_prediction_baseline = val_wt_groundtruth_baseline(wt_test_prediction, test_loader)

            all_mse_wt_groundtruth_baseline, all_rmse_wt_groundtruth_baseline, all_rp_wt_groundtruth_baseline, all_cindex_wt_groundtruth_baseline = val_wt_groundtruth_baseline(wt_all_affinity, all_loader)
            _, _, _, _, wt_all_prediction, _ = val_ensemble(models_original, wt_all_loader, device)
            all_mse_wt_prediction_baseline, all_rmse_wt_prediction_baseline, all_rp_wt_prediction_baseline, all_cindex_wt_prediction_baseline = val_wt_groundtruth_baseline(wt_all_prediction, all_loader)

            
        
            if args_.split_method == 'different_mutation_different_drug':
                ratio_groundtruth = (affinity_mut1_drug2 * affinity_mut2_drug1) / affinity_mut1_drug1
                test_mse_ratio_groundtruth_baseline, test_rmse_ratio_groundtruth_baseline, test_rp_ratio_groundtruth_baseline = val_wt_groundtruth_baseline(ratio_groundtruth, test_loader)
                _, _, _, _, mut_lig_prediction, _ = val_ensemble(models_original, train_loader, device)
                ratio_prediction = (mut_lig_prediction[1] * mut_lig_prediction[2]) / mut_lig_prediction[0]
                test_mse_ratio_prediction_baseline, test_rmse_ratio_prediction_baseline, test_rp_ratio_prediction_baseline = val_wt_groundtruth_baseline(ratio_prediction, test_loader)

            test_mse_original, test_rmse_original, test_rp_original, test_cindex_original, pred_original, label = val_ensemble(models_original, test_loader, device)
            test_mse_finetuning, test_rmse_finetuning, test_rp_finetuning, test_cindex_finetuning, pred_finetuning, label = val_ensemble(models_finetuning, test_loader, device)
            all_mse_original, all_rmse_original, all_rp_original, all_cindex_original, pred_original_all, label_all = val_ensemble(models_original, all_loader, device)
            

            if args_.split_method == 'different_mutation_same_drug':
                print(f'label: {label}')
                print(f'gt_wt: {wt_all_affinity}')
                print(f'prediction_wt: {wt_all_prediction}')
                print(f'prediction_original: {pred_original}')
                print(f'prediction_finetuning: {pred_finetuning}')

            elif args_.split_method == 'same_mutation_different_drug':
                print(f'label: {label}')
                print(f'gt_wt: {wt_test_affinity}')
                print(f'prediction_wt: {wt_test_prediction}')
                print(f'prediction_original: {pred_original}')
                print(f'prediction_finetuning: {pred_finetuning}')

            elif args_.split_method == 'different_mutation_different_drug':
                print(f'label: {label}')
                print(f'gt_wt: {wt_affinity}')
                print(f'prediction_wt: {wt_prediction}')
                print(f'gt_ratio: {ratio_groundtruth}')
                print(f'prediction_ratio: {ratio_prediction}')
                print(f'prediction_original: {pred_original}')
                print(f'prediction_finetuning: {pred_finetuning}')
            
            else:
                raise ValueError('split_method and combination_seed are not matched')

            msg = f"model_seed: {model_seed}, test_mse_original: {test_mse_original:.4f}, test_rmse_original: {test_rmse_original:.4f}, test_rp_original: {test_rp_original:.4f}, test_cindex_original: {test_cindex_original:.4f},\
                    test_mse_finetuning: {test_mse_finetuning:.4f}, test_rmse_finetuning: {test_rmse_finetuning:.4f}, test_rp_finetuning: {test_rp_finetuning:.4f}, test_cindex_finetuning: {test_cindex_finetuning:.4f}"
            logger.info(msg)
            

            
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

            if args_.split_method == 'different_mutation_different_drug':
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

        if args_.split_method == 'different_mutation_same_drug':
            all_drug_type.append(drug_type)
            all_drug_name.append(drug_name)
            all_train_num.append(train_num)
            all_test_num.append(test_num)
        elif args_.split_method == 'same_mutation_different_drug':
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

        if args_.split_method == 'different_mutation_different_drug':
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

    
    if args_.split_method == 'different_mutation_same_drug':
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
    elif args_.split_method == 'same_mutation_different_drug':
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
    else:
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
   
    df.to_csv(f'result_finetuning_{args_.split_method}_epoch{args_.epochs}_lr{args_.lr}_combinationseed{combination_seed}.csv', index=False)