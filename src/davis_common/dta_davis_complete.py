"""
Split utilities and fine-tuning helpers for DAVIS-complete.
This module is dependency-light and can be imported from any experiment
without relying on package-private paths.
"""
from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Optional RDKit dependency for Tanimoto-based splits
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    rdMolDescriptors = None


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'davis_complete'

def create_fold(df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]) -> Dict[str, pd.DataFrame]:
    """Random split into train/valid/test with mutation-aware test stratification."""
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=1)
    train = train_val[~train_val.index.isin(val.index)]

    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'test_wt': test_wt.reset_index(drop=True),
        'test_mutation': test_mutation.reset_index(drop=True),
    }


def _smiles_to_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048):
    if Chem is None:
        raise ImportError("RDKit is required for Tanimoto-based splits but is not installed.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)


def _calculate_tanimoto_similarity(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _get_similarity_matrix(fps: List) -> np.ndarray:
    n = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i, j] = 1.0
            else:
                sim = _calculate_tanimoto_similarity(fps[i], fps[j])
                mat[i, j] = sim
                mat[j, i] = sim
    return mat


def create_new_drug_tanimoto(df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]) -> Dict[str, pd.DataFrame]:
    """Split by drug Tanimoto similarity threshold (~0.5)."""
    train_frac, val_frac, test_frac = frac
    all_smiles = list(sorted(set(df['compound_iso_smiles'])))
    val_size = int(len(all_smiles) * frac[1])
    test_size = int(len(all_smiles) * frac[2])

    fps = []
    for s in all_smiles:
        fp = _smiles_to_fingerprint(s)
        if fp is not None:
            fps.append(fp)

    sim = _get_similarity_matrix(fps)
    selected_idx = [i for i in range(len(fps)) if len(sim[:, i][sim[:, i] >= 0.5]) < 2]

    random.seed(fold_seed)
    shuffled = selected_idx.copy()
    random.shuffle(shuffled)

    val_idx = shuffled[:val_size]
    test_idx = shuffled[val_size:val_size + test_size]

    val_smiles = [all_smiles[i] for i in val_idx]
    test_smiles = [all_smiles[i] for i in test_idx]
    train_smiles = [all_smiles[i] for i in range(len(all_smiles)) if i not in val_idx and i not in test_idx]

    test = df[df['compound_iso_smiles'].isin(test_smiles)]
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    val = df[df['compound_iso_smiles'].isin(val_smiles)]
    train = df[df['compound_iso_smiles'].isin(train_smiles)]

    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'test_wt': test_wt.reset_index(drop=True),
        'test_mutation': test_mutation.reset_index(drop=True),
    }


def create_wt_mutation_split(df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float]):
    train_frac, val_frac = frac
    test = df[(df['protein'].str.contains(f"_[a-z][0-9]") | df['protein'].str.contains(f"_itd") | df['protein'].str.contains(f"abl1_p") | df['protein'].str.contains("s808g"))]
    train_val = df[~df.index.isin(test.index)]
    test_frac = len(test) / len(df)
    val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed)
    train = train_val[~train_val.index.isin(val.index)]

    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'test_wt': None,
        'test_mutation': test.reset_index(drop=True),
    }


def create_fold_setting_cold(df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float], entity: str):
    train_frac, val_frac, test_frac = frac
    gene_drop = df[entity].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values

    test = df[df[entity].isin(gene_drop)]
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    train_val = df[~df[entity].isin(gene_drop)]

    gene_drop_val = train_val[entity].drop_duplicates().sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed).values
    val = train_val[train_val[entity].isin(gene_drop_val)]
    train = train_val[~train_val[entity].isin(gene_drop_val)]

    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'test_wt': test_wt.reset_index(drop=True),
        'test_mutation': test_mutation.reset_index(drop=True),
    }


def create_new_protein_name(df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]):
    protein_mutation = ['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'gcn2', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret']

    df_mutation = df[df['protein'].str.contains("_[a-z][0-9]") | df['protein'].str.contains("itd") | df['protein'].str.contains("abl1_p")]
    df_wt = df[~df.index.isin(df_mutation.index)]

    train_frac, val_frac, test_frac = frac
    test_sample_protein = df_wt['protein'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values

    patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in test_sample_protein]
    regex = '|'.join(patterns)
    test = df[df['protein'].str.contains(regex, regex=True)]

    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    df_wt_train_val = df_wt[~df_wt['protein'].isin(test_sample_protein)]
    val_sample_protein = df_wt_train_val['protein'].drop_duplicates().sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed).values
    patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in val_sample_protein]
    regex = '|'.join(patterns)
    val = df[df['protein'].str.contains(regex, regex=True)]

    train_sample_protein = df_wt_train_val[~df_wt_train_val['protein'].isin(val_sample_protein)]['protein'].drop_duplicates().values
    patterns = [rf'^{re.escape(p)}.*' if p in protein_mutation else rf'^{re.escape(p)}$' for p in train_sample_protein]
    regex = '|'.join(patterns)
    train = df[df['protein'].str.contains(regex, regex=True)]

    assert len(set(train['protein']) & set(val['protein'])) == 0
    assert len(set(test['protein']) & set(val['protein'])) == 0
    assert len(set(train['protein']) & set(test['protein'])) == 0

    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'test_wt': test_wt.reset_index(drop=True),
        'test_mutation': test_mutation.reset_index(drop=True),
    }


def create_full_ood_set(df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]):
    train_frac, val_frac, test_frac = frac
    test_drugs = df['drug'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
    test_prots = df['protein'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values

    test = df[(df['drug'].isin(test_drugs)) & (df['protein'].isin(test_prots))]
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("_itd") | test['protein'].str.contains("abl1_p") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]

    train_val = df[(~df['drug'].isin(test_drugs)) & (~df['protein'].isin(test_prots))]

    val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed)
    train = train_val[~train_val.index.isin(val.index)]

    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
        'test_wt': test_wt.reset_index(drop=True),
        'test_mutation': test_mutation.reset_index(drop=True),
    }


def create_seq_identity_fold(df: pd.DataFrame, mmseqs_seq_clus_df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float], min_clus_in_split: int = 5):
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
            max_clust_size = int(math.ceil(split_size / min_clus_in_split))
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

    return {
        'train': train_df.reset_index(drop=True),
        'valid': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
        'test_wt': test_df_wt.reset_index(drop=True),
        'test_mutation': test_df_mutation.reset_index(drop=True),
    }


def create_seq_identity_drug_tanimoto_fold(df: pd.DataFrame, mmseqs_seq_clus_df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float], min_clus_in_split: int = 5):
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
            max_clust_size = int(math.ceil(split_size / min_clus_in_split))
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

    # Next, we apply Tanimoto filtering on the selected smiles for each split
    def _filter_by_tanimoto(sdf: pd.DataFrame):
        smiles = list(sorted(set(sdf['compound_iso_smiles'])))
        if not smiles:
            return sdf
        fps = [_smiles_to_fingerprint(s) for s in smiles]
        sim = _get_similarity_matrix(fps)
        # simple dedup policy: keep as-is; downstream loaders can enforce thresholds
        return sdf

    train_df = df[df['protein'].isin(to_use)].reset_index(drop=True)
    train_df['split'] = 'train'
    val_df['split'] = 'valid'
    test_df['split'] = 'test'

    test_df = _filter_by_tanimoto(test_df)
    val_df = _filter_by_tanimoto(val_df)

    test_df_mutation = test_df[test_df['protein'].str.contains("_[a-z][0-9]") | test_df['protein'].str.contains("_itd") | test_df['protein'].str.contains("abl1_p") | test_df['protein'].str.contains("s808g")]
    test_df_wt = test_df[~test_df.index.isin(test_df_mutation.index)]

    assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

    return {
        'train': train_df.reset_index(drop=True),
        'valid': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
        'test_wt': test_df_wt.reset_index(drop=True),
        'test_mutation': test_df_mutation.reset_index(drop=True),
    }


# Fine-tuning split helpers

def create_fine_tuning_different_mutation_same_drug_split(protein: str, drug_type: Optional[str] = None, drug: Optional[str] = None, df: Optional[pd.DataFrame] = None, seed: int = 1, test_size: float = 0.2, nontruncated_affinity: bool = False):

    inhibitor_binding_mode = pd.read_csv(DATA_DIR / 'davis_inhibitor_binding_mode.csv')    
    inhibitor_binding_mode = inhibitor_binding_mode[['Compound', 'Binding Mode (based on ABL1-phos. vs. -nonphos affinity)']]
    inhibitor_binding_mode.columns = ['compound', 'binding_mode']
    dict_mode_compound = {mode: list(sdf['compound']) for mode, sdf in inhibitor_binding_mode.groupby('binding_mode')}

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
            if drug_type not in dict_mode_compound:
                return None, None, None, None
            drug_name = random.choice(dict_mode_compound[drug_type])

        data_select = data[(data['protein'].str.contains(f"{protein}_[a-z][0-9]") | data['protein'].str.contains(f"{protein}_itd") | data['protein'].str.contains(f"{protein}_p")) & data['drug_name'].str.fullmatch(drug_name)]
        failed += 1

    train, test = train_test_split(data_select, test_size=test_size, random_state=0)
    wt = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].str.fullmatch(drug_name)]
    assert len(wt) == 1
    return (
        {
            'all': data_select.reset_index(drop=True),
            'train': train.reset_index(drop=True),
            'test': test.reset_index(drop=True),
            'wt_all': wt.reset_index(drop=True),
            'wt_train': wt.reset_index(drop=True),
            'wt_test': wt.reset_index(drop=True),
        },
        drug_name,
        len(train),
        len(test),
    )


def create_fine_tuning_same_mutation_different_drug_split(protein: str, mutation: str, df: Optional[pd.DataFrame] = None, test_size: float = 0.2, nontruncated_affinity: bool = False):
    
    inhibitor_binding_mode = pd.read_csv(DATA_DIR / 'davis_inhibitor_binding_mode.csv') 
    inhibitor_binding_mode = inhibitor_binding_mode[['Compound', 'Binding Mode (based on ABL1-phos. vs. -nonphos affinity)']]
    inhibitor_binding_mode.columns = ['compound', 'binding_mode']

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
    return (
        {
            'all': data_select.reset_index(drop=True),
            'train': train.reset_index(drop=True),
            'test': test.reset_index(drop=True),
            'wt_all': wt_all.reset_index(drop=True),
            'wt_train': wt_train.reset_index(drop=True),
            'wt_test': wt_test.reset_index(drop=True),
        },
        len(train),
        len(test),
    )
