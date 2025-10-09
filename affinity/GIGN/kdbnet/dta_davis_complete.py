"""
Drug-target binding affinity datasets
"""
import math
import yaml
import json
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from kdbnet import pdb_graph, mol_graph
from typing import Literal
import random
from sklearn.model_selection import train_test_split
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pickle
import re
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from typing import List, Tuple, Set


RDLogger.DisableLog('rdApp.*')

class DTA(data.Dataset):
    """
    Base class for loading drug-target binding affinity datasets.
    Adapted from: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
    """
    def __init__(self, df=None, data_list=None, onthefly=False,
                prot_featurize_fn=None, drug_featurize_fn=None):
        """
        Parameters
        ----------
            df : pd.DataFrame with columns [`drug`, `protein`, `y`],
                where `drug`: drug key, `protein`: protein key, `y`: binding affinity.
            data_list : list of dict (same order as df)
                if `onthefly` is True, data_list has the PDB coordinates and SMILES strings
                    {`drug`: SDF file path, `protein`: coordinates dict (`pdb_data` in `DTATask`), `y`: float}
                if `onthefly` is False, data_list has the cached torch_geometric graphs
                    {`drug`: `torch_geometric.data.Data`, `protein`: `torch_geometric.data.Data`, `y`: float}
                `protein` has attributes:
                    -x          alpha carbon coordinates, shape [n_nodes, 3]
                    -edge_index edge indices, shape [2, n_edges]
                    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
                    -name       name of the protein structure, string
                    -node_s     node scalar features, shape [n_nodes, 6]
                    -node_v     node vector features, shape [n_nodes, 3, 3]
                    -edge_s     edge scalar features, shape [n_edges, 39]
                    -edge_v     edge scalar features, shape [n_edges, 1, 3]
                    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
                    -seq_emb    sequence embedding (ESM1b), shape [n_nodes, 1280]
                `drug` has attributes:
                    -x          atom coordinates, shape [n_nodes, 3]
                    -edge_index edge indices, shape [2, n_edges]
                    -node_s     node scalar features, shape [n_nodes, 66]
                    -node_v     node vector features, shape [n_nodes, 1, 3]
                    -edge_s     edge scalar features, shape [n_edges, 16]
                    -edge_v     edge scalar features, shape [n_edges, 1, 3]
                    -name       name of the drug, string
            onthefly : bool
                whether to featurize data on the fly or pre-compute
            prot_featurize_fn : function
                function to featurize a protein.
            drug_featurize_fn : function
                function to featurize a drug.
        """
        super(DTA, self).__init__()
        self.data_df = df
        self.data_list = data_list
        self.onthefly = onthefly
        if onthefly:
            assert prot_featurize_fn is not None, 'prot_featurize_fn must be provided'
            assert drug_featurize_fn is not None, 'drug_featurize_fn must be provided'
        self.prot_featurize_fn = prot_featurize_fn
        self.drug_featurize_fn = drug_featurize_fn

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.onthefly:
            drug = self.drug_featurize_fn(
                self.data_list[idx]['drug'],
                name=self.data_list[idx]['drug_name']
            )
            prot = self.prot_featurize_fn(
                self.data_list[idx]['protein'],
                name=self.data_list[idx]['protein_name']
            )
        else:
            drug = self.data_list[idx]['drug']
            prot = self.data_list[idx]['protein']
        y = self.data_list[idx]['y']
        item = {'drug': drug, 'protein': prot, 'y': y}
        return item


def create_fold(df, fold_seed, frac):
    """
    Create train/valid/test folds by random splitting.
    Adapted from: https://github.com/mims-harvard/TDC/blob/2d4fb74ac00e88986306b2b12ffdb3be87418719/tdc/utils.py#L375
    """
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]
    
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True),
            'test_wt': test_wt.reset_index(drop = True),
            'test_mutation': test_mutation.reset_index(drop = True)}



def smiles_to_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048):
        """Convert SMILES to Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)

def calculate_tanimoto_similarity(fp1, fp2):
    """Calculate Tanimoto similarity between two fingerprints"""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_similarity_matrix(fingerprints: List):
    """Calculate pairwise Tanimoto similarity matrix"""
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                sim = calculate_tanimoto_similarity(fingerprints[i], fingerprints[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    
    return similarity_matrix


def create_new_drug_tanimoto(df, fold_seed, frac):
    """
    Create train/valid/test folds by tanimoto similarity of drugs. Similarity of 0.5 is used as the threshold.
    """
    
    train_frac, val_frac, test_frac = frac
    all_smiles = list(sorted(set(df['compound_iso_smiles'])))
    val_size = int(len(all_smiles) * frac[1])
    test_size = int(len(all_smiles) * frac[2])

    all_fp = []
    for smiles in all_smiles:
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            all_fp.append(fp)

    similarity_matrix_all_fp = get_similarity_matrix(all_fp)

    selected_index = []
    for i in range(len(all_fp)):
        if len(similarity_matrix_all_fp[:, i][similarity_matrix_all_fp[:, i] >= 0.5]) < 2:
            selected_index.append(i)
           
            
    random.seed(fold_seed)
    shuffled_indices = selected_index.copy()
    random.shuffle(shuffled_indices)

    val_indices = shuffled_indices[:val_size]
    test_indices = shuffled_indices[val_size:val_size + test_size]

    val_smiles = [all_smiles[i] for i in val_indices]
    test_smiles = [all_smiles[i] for i in test_indices]
    train_smiles = [all_smiles[i] for i in range(len(all_smiles)) if i not in val_indices and i not in test_indices]
    
    test = df[df['compound_iso_smiles'].isin(test_smiles)]
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    val = df[df['compound_iso_smiles'].isin(val_smiles)]
    train = df[df['compound_iso_smiles'].isin(train_smiles)]

    return {'train': train.reset_index(drop=True),
            'valid': val.reset_index(drop=True),
            'test': test.reset_index(drop=True),
            'test_wt': test_wt.reset_index(drop=True),
            'test_mutation': test_mutation.reset_index(drop=True)}
    
    

def create_wt_mutation_split(df, fold_seed, frac):
    train_frac, val_frac = frac
    test = df[(df['protein'].str.contains(f"_[a-z][0-9]") | df['protein'].str.contains(f"_itd") | df['protein'].str.contains(f"abl1_p") | df['protein'].str.contains("s808g"))]
    train_val = df[~df.index.isin(test.index)]
    test_frac = len(test) / len(df)
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed)
    train = train_val[~train_val.index.isin(val.index)]

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True),
            'test_wt': None,
            'test_mutation': test.reset_index(drop = True)}


def create_fold_setting_cold(df, fold_seed, frac, entity):
    """
    Create train/valid/test folds by drug/protein-wise splitting.
    Adapted from: https://github.com/mims-harvard/TDC/blob/2d4fb74ac00e88986306b2b12ffdb3be87418719/tdc/utils.py#L388
    """
    train_frac, val_frac, test_frac = frac
    gene_drop = df[entity].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values

    test = df[df[entity].isin(gene_drop)]
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    train_val = df[~df[entity].isin(gene_drop)]

    gene_drop_val = train_val[entity].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    val = train_val[train_val[entity].isin(gene_drop_val)]
    train = train_val[~train_val[entity].isin(gene_drop_val)]

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True),
            'test_wt': test_wt.reset_index(drop = True),
            'test_mutation': test_mutation.reset_index(drop = True)}



def create_new_protein_name(df, fold_seed, frac):
    """
    Create train/valid/test folds by drug/protein-wise splitting.
    Adapted from: https://github.com/mims-harvard/TDC/blob/2d4fb74ac00e88986306b2b12ffdb3be87418719/tdc/utils.py#L388
    """
    protein_mutation = ['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'gcn2', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'] 

    df_mutation = df[df['protein'].str.contains("_[a-z][0-9]") | df['protein'].str.contains("itd") | df['protein'].str.contains("abl1_p")]
    df_wt = df[~df.index.isin(df_mutation.index)]

    train_frac, val_frac, test_frac = frac
    test_sample_protein = df_wt['protein'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values

    patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in test_sample_protein]
    regex = '|'.join(patterns)
    test = df[df['protein'].str.contains(regex, regex=True)]

    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    df_wt_train_val = df_wt[~df_wt['protein'].isin(test_sample_protein)]
    val_sample_protein = df_wt_train_val['protein'].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in val_sample_protein]
    regex = '|'.join(patterns)
    val = df[df['protein'].str.contains(regex, regex=True)]

    train_sample_protein = df_wt_train_val[~df_wt_train_val['protein'].isin(val_sample_protein)]['protein'].drop_duplicates().values
    patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in train_sample_protein]
    regex = '|'.join(patterns)
    train = df[df['protein'].str.contains(regex, regex=True)]
    
    # Verify no overlap between splits
    assert len(set(train['protein']) & set(val['protein'])) == 0, "Overlap between train and validation proteins"
    assert len(set(test['protein']) & set(val['protein'])) == 0, "Overlap between test and validation proteins"
    assert len(set(train['protein']) & set(test['protein'])) == 0, "Overlap between train and test proteins"

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True),
            'test_wt': test_wt.reset_index(drop = True),
            'test_mutation': test_mutation.reset_index(drop = True)}


def create_full_ood_set(df, fold_seed, frac):
    """
    Create train/valid/test folds such that drugs and proteins are
    not overlapped in train and test sets. Train and valid may share
    drugs and proteins (random split).
    """
    train_frac, val_frac, test_frac = frac
    test_drugs = df['drug'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
    test_prots = df['protein'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values

    test = df[(df['drug'].isin(test_drugs)) & (df['protein'].isin(test_prots))]
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]


    train_val = df[(~df['drug'].isin(test_drugs)) & (~df['protein'].isin(test_prots))]

    val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed)
    train = train_val[~train_val.index.isin(val.index)]

    return {'train': train.reset_index(drop=True),
            'valid': val.reset_index(drop=True),
            'test': test.reset_index(drop=True),
            'test_wt': test_wt.reset_index(drop=True),
            'test_mutation': test_mutation.reset_index(drop=True)}


def create_seq_identity_fold(df, mmseqs_seq_clus_df, fold_seed, frac, min_clus_in_split=5):
    """
    Adapted from: https://github.com/drorlab/atom3d/blob/master/atom3d/splits/sequence.py
    Clusters are selected randomly into validation and test sets,
    but to ensure that there is some diversity in each set
    (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced.
    Some data examples may be removed in order to satisfy this constraint.
    """
    _rng = np.random.RandomState(fold_seed)

    def _parse_mmseqs_cluster_res(mmseqs_seq_clus_df):
        clus2seq, seq2clus = {}, {}
        for rep, sdf in mmseqs_seq_clus_df.groupby('rep'):
            for seq in sdf['seq']:
                if rep not in clus2seq:
                    clus2seq[rep] = []
                clus2seq[rep].append(seq)
                seq2clus[seq] = rep
        return seq2clus, clus2seq

    def _create_cluster_split(df, seq2clus, clus2seq, to_use, split_size, min_clus_in_split):
        data = df.copy()
        all_prot = set(seq2clus.keys())
        used = all_prot.difference(to_use)
        split = None
        while True:
            p = _rng.choice(sorted(to_use))
            c = seq2clus[p]
            members = set(clus2seq[c])
            members = members.difference(used)
            if len(members) == 0:
                continue
            # ensure that at least min_fam_in_split families in each split
            max_clust_size = int(np.ceil(split_size / min_clus_in_split))
            sel_prot = list(members)[:max_clust_size]
            sel_df = data[data['protein'].isin(sel_prot)]
            split = sel_df if split is None else pd.concat([split, sel_df])
            to_use = to_use.difference(members)
            used = used.union(members)
            if len(split) >= split_size:
                break
        split = split.reset_index(drop=True)
        return split, to_use

    seq2clus, clus2seq = _parse_mmseqs_cluster_res(mmseqs_seq_clus_df)
    train_frac, val_frac, test_frac = frac
    test_size, val_size = len(df) * test_frac, len(df) * val_frac
    to_use = set(seq2clus.keys())

    val_df, to_use = _create_cluster_split(df, seq2clus, clus2seq, to_use, val_size, min_clus_in_split)
    test_df, to_use = _create_cluster_split(df, seq2clus, clus2seq, to_use, test_size, min_clus_in_split)
    train_df = df[df['protein'].isin(to_use)].reset_index(drop=True)
    train_df['split'] = 'train'
    val_df['split'] = 'valid'
    test_df['split'] = 'test'

    test_df_mutation = test_df[test_df['protein'].str.contains("_[a-z][0-9]") | test_df['protein'].str.contains("_itd") | test_df['protein'].str.contains("abl1_p") | test_df['protein'].str.contains("s808g")]
    test_df_wt = test_df[~test_df.index.isin(test_df_mutation.index)]

    assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

    return {'train': train_df.reset_index(drop=True),
            'valid': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True),
            'test_wt': test_df_wt.reset_index(drop=True),
            'test_mutation': test_df_mutation.reset_index(drop=True)}


def create_seq_identity_drug_tanimoto_fold(df, mmseqs_seq_clus_df, fold_seed, frac, min_clus_in_split=5):
    """
    Adapted from: https://github.com/drorlab/atom3d/blob/master/atom3d/splits/sequence.py
    Clusters are selected randomly into validation and test sets,
    but to ensure that there is some diversity in each set
    (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced.
    Some data examples may be removed in order to satisfy this constraint.
    """
    _rng = np.random.RandomState(fold_seed)

    def _parse_mmseqs_cluster_res(mmseqs_seq_clus_df):
        clus2seq, seq2clus = {}, {}
        for rep, sdf in mmseqs_seq_clus_df.groupby('rep'):
            for seq in sdf['seq']:
                if rep not in clus2seq:
                    clus2seq[rep] = []
                clus2seq[rep].append(seq)
                seq2clus[seq] = rep
        return seq2clus, clus2seq

    def _create_cluster_split(df, seq2clus, clus2seq, to_use, split_size, min_clus_in_split):
        data = df.copy()
        all_prot = set(seq2clus.keys())
        used = all_prot.difference(to_use)
        split = None
        while True:
            p = _rng.choice(sorted(to_use))
            c = seq2clus[p]
            members = set(clus2seq[c])
            members = members.difference(used)
            if len(members) == 0:
                continue
            # ensure that at least min_fam_in_split families in each split
            max_clust_size = int(np.ceil(split_size / min_clus_in_split))
            sel_prot = list(members)[:max_clust_size]
            sel_df = data[data['protein'].isin(sel_prot)]
            split = sel_df if split is None else pd.concat([split, sel_df])
            to_use = to_use.difference(members)
            used = used.union(members)
            if len(split) >= split_size:
                break
        split = split.reset_index(drop=True)
        return split, to_use

    seq2clus, clus2seq = _parse_mmseqs_cluster_res(mmseqs_seq_clus_df)
    train_frac, val_frac, test_frac = frac
    test_size, val_size = len(df) * test_frac, len(df) * val_frac
    to_use = set(seq2clus.keys())

    val_df, to_use = _create_cluster_split(df, seq2clus, clus2seq, to_use, val_size, min_clus_in_split)
    test_df, to_use = _create_cluster_split(df, seq2clus, clus2seq, to_use, test_size, min_clus_in_split)
    train_df = df[df['protein'].isin(to_use)].reset_index(drop=True)
    train_df['split'] = 'train'
    val_df['split'] = 'valid'
    test_df['split'] = 'test'
    
    ### here we add the drug tanimoto similarity filtering
    train_frac, val_frac, test_frac = frac
    all_smiles = list(sorted(set(df['compound_iso_smiles'])))
    val_size = int(len(all_smiles) * frac[1])
    test_size = int(len(all_smiles) * frac[2])

    all_fp = []
    for smiles in all_smiles:
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            all_fp.append(fp)

    similarity_matrix_all_fp = get_similarity_matrix(all_fp)

    selected_index = []
    for i in range(len(all_fp)):
        if len(similarity_matrix_all_fp[:, i][similarity_matrix_all_fp[:, i] >= 0.5]) < 2:
            selected_index.append(i)
           
            
    random.seed(fold_seed)
    shuffled_indices = selected_index.copy()
    random.shuffle(shuffled_indices)

    val_indices = shuffled_indices[:val_size]
    test_indices = shuffled_indices[val_size:val_size + test_size]

    val_smiles = [all_smiles[i] for i in val_indices]
    test_smiles = [all_smiles[i] for i in test_indices]
    train_smiles = [all_smiles[i] for i in range(len(all_smiles)) if i not in val_indices and i not in test_indices]
    ####
    
    test_df = test_df[test_df['compound_iso_smiles'].isin(test_smiles)]
    val_df = val_df[val_df['compound_iso_smiles'].isin(val_smiles)]
    train_df = train_df[train_df['compound_iso_smiles'].isin(train_smiles)]
    

    test_df_mutation = test_df[test_df['protein'].str.contains("_[a-z][0-9]") | test_df['protein'].str.contains("_itd") | test_df['protein'].str.contains("abl1_p") | test_df['protein'].str.contains("s808g")]
    test_df_wt = test_df[~test_df.index.isin(test_df_mutation.index)]

    assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

    return {'train': train_df.reset_index(drop=True),
            'valid': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True),
            'test_wt': test_df_wt.reset_index(drop=True),
            'test_mutation': test_df_mutation.reset_index(drop=True)}


#TODO: add the path of davis_inhibitor_binding_mode.csv as a argument

def create_fine_tuning_different_mutation_same_drug_split(protein, drug_type=None, drug=None, df=None, seed=1, test_size=0.2, nontruncated_affinity=False):
    try:
        inhibitor_binding_mode = pd.read_csv('/data/mwu11/DAVIS-complete/data/davis_complete/davis_inhibitor_binding_mode.csv')
    except FileNotFoundError:
        inhibitor_binding_mode = pd.read_csv('../data/davis_complete/davis_inhibitor_binding_mode.csv')
    except Exception as e:
        raise e

    inhibitor_binding_mode = inhibitor_binding_mode[['Compound', 'Binding Mode (based on ABL1-phos. vs. -nonphos affinity)']]
    inhibitor_binding_mode.columns = ['compound', 'binding_mode']
    dict_mode_compound = {mode: list(df['compound']) for mode, df in inhibitor_binding_mode.groupby('binding_mode')}
    
    data = df.copy()
    if nontruncated_affinity:
        data = data[data['y'] > 5.0]

    random.seed(seed)
    data_select = pd.DataFrame()
    failed = 0

    while len(data_select) <= 2:
        if failed > 72:
            return None, None, None, None
        if drug:
            drug_name = drug
        else:
            drug_name = random.choice(dict_mode_compound[drug_type])

        data_select = data[(data['protein'].str.contains(f"{protein}_[a-z][0-9]") | data['protein'].str.contains(f"{protein}_itd") | data['protein'].str.contains(f"{protein}_p")) & data['drug_name'].str.fullmatch(drug_name)]
        failed += 1

    train, test = train_test_split(data_select, test_size=test_size, random_state=0)
    wt = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].str.fullmatch(drug_name)]
    assert len(wt) == 1
    return {'all': data_select.reset_index(drop=True), 'train': train.reset_index(drop = True), 'test': test.reset_index(drop = True), 'wt_all': wt.reset_index(drop = True), 'wt_train': wt.reset_index(drop = True), 'wt_test': wt.reset_index(drop = True)}, drug_name, len(train), len(test)


#TODO: add the path of davis_inhibitor_binding_mode.csv as a argument

def create_fine_tuning_same_mutation_different_drug_split(protein, mutation, df=None, test_size=0.2, nontruncated_affinity=False):
    try:
        inhibitor_binding_mode = pd.read_csv('/data/mwu11/DAVIS-complete/data/davis_complete/davis_inhibitor_binding_mode.csv')
    except FileNotFoundError:
        inhibitor_binding_mode = pd.read_csv('../data/davis_complete/davis_inhibitor_binding_mode.csv')
    except Exception as e:
        raise e
    inhibitor_binding_mode = inhibitor_binding_mode[['Compound', 'Binding Mode (based on ABL1-phos. vs. -nonphos affinity)']]
    inhibitor_binding_mode.columns = ['compound', 'binding_mode']
    dict_mode_compound = {mode: list(df['compound']) for mode, df in inhibitor_binding_mode.groupby('binding_mode')}
    
    data = df.copy()
    if nontruncated_affinity:
        data = data[data['y'] > 5.0]

    data_select = data[data['protein'].str.fullmatch(mutation)].sort_values(by='drug_name')
    if len(data_select) <= 2:
        return None, None, None, None
      
    train, test = train_test_split(data_select, test_size=test_size, random_state=0)
    train = train.sort_values(by='drug_name')
    test = test.sort_values(by='drug_name')
    wt_all = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].isin(data_select['drug_name'].unique())].sort_values(by='drug_name')
    wt_train = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].isin(train['drug_name'].unique())].sort_values(by='drug_name')
    wt_test = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].isin(test['drug_name'].unique())].sort_values(by='drug_name')
    
    assert len(data_select) == len(wt_all)
    assert len(train) == len(wt_train)
    assert len(test) == len(wt_test)
    return {'all': data_select.reset_index(drop=True), 'train': train.reset_index(drop=True), 'test': test.reset_index(drop=True), 'wt_all': wt_all.reset_index(drop=True), 'wt_train': wt_train.reset_index(drop=True), 'wt_test': wt_test.reset_index(drop=True)}, len(train), len(test)

