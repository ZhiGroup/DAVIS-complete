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
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("itd") | test['protein'].str.contains("s808g")]
    test_wt = test[~test.index.isin(test_mutation.index)]
    
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True),
            'test_wt': test_wt.reset_index(drop = True),
            'test_mutation': test_mutation.reset_index(drop = True)}


def create_wt_mutation_split(df, fold_seed, frac):
    train_frac, val_frac = frac
    #TODO: should include abl1_p as now we define modification which include phosphorylation
    test = df[df['protein'].str.contains("_[a-z][0-9]") | df['protein'].str.contains("itd") | df['protein'].str.contains("s808g")]
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
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("itd") | test['protein'].str.contains("s808g")]
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
    test_mutation = test[test['protein'].str.contains("_[a-z][0-9]") | test['protein'].str.contains("itd") | test['protein'].str.contains("s808g")]
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

    test_df_mutation = test_df[test_df['protein'].str.contains("_[a-z][0-9]") | test_df['protein'].str.contains("itd") | test_df['protein'].str.contains("s808g")]
    test_df_wt = test_df[~test_df.index.isin(test_df_mutation.index)]

    assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

    return {'train': train_df.reset_index(drop=True),
            'valid': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True),
            'test_wt': test_df_wt.reset_index(drop=True),
            'test_mutation': test_df_mutation.reset_index(drop=True)}


def create_fine_tuning_different_mutation_same_drug_split(protein, drug_type=None, drug=None, df=None, seed=1, test_size=0.2, nontruncated_affinity=False):

    inhibitor_binding_mode = pd.read_csv('/data/mwu11/FDA/data/davis_complete/davis_inhibitor_binding_mode.csv')
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
        #data_select = data[(data['protein'].str.contains(f"{protein}_[a-z][0-9]") | data['protein'].str.contains(f"{protein}_itd")) & data['drug_name'].str.fullmatch(drug_name)]
        failed += 1

    train, test = train_test_split(data_select, test_size=test_size, random_state=0)
    wt = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].str.fullmatch(drug_name)]
    assert len(wt) == 1
    return {'all': data_select.reset_index(drop=True), 'train': train.reset_index(drop = True), 'test': test.reset_index(drop = True), 'wt_all': wt.reset_index(drop = True), 'wt_train': wt.reset_index(drop = True), 'wt_test': wt.reset_index(drop = True)}, drug_name, len(train), len(test)

def create_fine_tuning_same_mutation_different_drug_split(protein, mutation, df=None, test_size=0.2, nontruncated_affinity=False):
    inhibitor_binding_mode = pd.read_csv('/data/mwu11/FDA/data/davis_complete/davis_inhibitor_binding_mode.csv')
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



#%%
def create_fine_tuning_different_mutation_different_drug_split(protein: Literal['abl1', 'egfr', 'flt3', 'kit', 'met', 'pik3ca', 'ret'], drug_1_type=None, drug_2_type=None, drug_1=None, drug_2=None, df=None, seed=1, nontruncated_affinity=False):
    inhibitor_binding_mode = pd.read_csv('/data/mwu11/FDA/data/davis_complete/davis_inhibitor_binding_mode.csv')
    inhibitor_binding_mode = inhibitor_binding_mode[['Compound', 'Binding Mode (based on ABL1-phos. vs. -nonphos affinity)']]
    inhibitor_binding_mode.columns = ['compound', 'binding_mode']
    dict_mode_compound = {mode: list(df['compound']) for mode, df in inhibitor_binding_mode.groupby('binding_mode')}

    data = df.copy()
    if nontruncated_affinity:
        data = data[data['y'] > 5.0]

    random.seed(seed)
    mut_1, mut_2, drug_1_name, drug_2_name = None, None, None, None
    data_select = data[data['protein'].isin([mut_1, mut_2]) & data['drug_name'].isin([drug_1_name, drug_2_name])]
    
    while len(data_select) < 4:
        if drug_1 and drug_2:
            drug_1_name, drug_2_name = drug_1, drug_2
        else:
            drug_1_name = random.choice(dict_mode_compound[drug_1_type])
            drug_2_name = random.choice(dict_mode_compound[drug_2_type])

            while drug_1_name == drug_2_name:
                drug_2_name = random.choice(dict_mode_compound[drug_2_type])

        # print(f"Selected drugs: {drug_1_name}, {drug_2_name}")

        mut_1 = random.choice(data[data['protein'].str.contains(f"{protein}_[a-z][0-9]")]['protein'].unique())
        mut_2 = random.choice(data[data['protein'].str.contains(f"{protein}_[a-z][0-9]")]['protein'].unique())

        while mut_1 == mut_2:
            mut_2 = random.choice(data[data['protein'].str.contains(f"{protein}_[a-z][0-9]")]['protein'].unique())

        # print(f"Selected mutations: {mut_1}, {mut_2}")
        
        data_select = data[data['protein'].isin([mut_1, mut_2]) & data['drug_name'].isin([drug_1_name, drug_2_name])]
    
    
    train_mut_1_drug_1 = data_select[(data_select['protein'] == mut_1) & (data_select['drug_name'] == drug_1_name)]
    train_mut_1_drug_2 = data_select[(data_select['protein'] == mut_1) & (data_select['drug_name'] == drug_2_name)]
    train_mut_2_drug_1 = data_select[(data_select['protein'] == mut_2) & (data_select['drug_name'] == drug_1_name)]
    
    train = pd.concat([train_mut_1_drug_1, train_mut_1_drug_2, train_mut_2_drug_1])
    test = data_select[(data_select['protein'] == mut_2) & (data_select['drug_name'] == drug_2_name)]
    wt = df[df['protein'].str.fullmatch(f"{protein}") & df['drug_name'].str.fullmatch(drug_2_name)]
    assert len(wt) == 1
    return {'all': data_select.reset_index(drop = True), 'train': train.reset_index(drop = True), 'test': test.reset_index(drop = True), 'wt': wt.reset_index(drop = True)}, mut_1, mut_2, drug_1_name, drug_2_name



class DTATask(object):
    """
    Drug-target binding task (e.g., KIBA or Davis).
    Three splits: train/valid/test, each split is a DTA() class
    """
    def __init__(self,
            task_name=None,
            df=None,
            prot_pdb_id=None, pdb_data=None,
            emb_dir=None,
            drug_sdf_dir=None,
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8.,
            split_method='random', split_frac=[0.7, 0.1, 0.2],
            mmseqs_seq_clus_df=None,
            seed=42, onthefly=False
        ):
        """
        Parameters
        ----------
        task_name: str
            Name of the task (e.g., KIBA, Davis, etc.)
        df: pd.DataFrame
            Dataframe containing the data
        prot_pdb_id: dict
            Dictionary mapping protein name to PDB ID
        pdb_data: dict
            A json format of pocket structure data, where key is the PDB ID
            and value is the corresponding PDB structure data in a dictionary:
                -'name': kinase name
                -'UniProt_id': UniProt ID
                -'PDB_id': PDB ID,
                -'chain': chain ID,
                -'seq': pocket sequence,                
                -'coords': coordinates of the 'N', 'CA', 'C', 'O' atoms of the pocket residues,
                    - "N": [[x, y, z], ...]
                    - "CA": [[], ...],
                    - "C": [[], ...],
                    - "O": [[], ...]               
            (there are some other keys but only for internal use)
        emb_dir: str
            Directory containing the protein embeddings
        drug_sdf_dir: str
            Directory containing the drug SDF files
        num_pos_emb: int
            Dimension of positional embeddings
        num_rbf: int
            Number of radial basis functions
        contact_cutoff: float
            Cutoff distance for defining residue-residue contacts
        split_method: str
            how to split train/test sets, 
            -`random`: random split
            -`protein`: split by protein
            -`drug`: split by drug
            -`both`: unseen drugs and proteins in test set
            -`seqid`: split by protein sequence identity 
                (need to priovide the MMseqs2 sequence cluster result,
                see `mmseqs_seq_clus_df`)
        split_frac: list
            Fraction of data in train/valid/test sets
        mmseqs_seq_clus_df: pd.DataFrame
            Dataframe containing the MMseqs2 sequence cluster result
            using a desired sequence identity cutoff
        seed: int
            Random seed
        onthefly: bool
            whether to featurize data on the fly or pre-compute
        """
        self.task_name = task_name        
        self.prot_pdb_id = prot_pdb_id
        self.pdb_data = pdb_data        
        self.emb_dir = emb_dir
        self.df = df
        self.prot_featurize_params = dict(
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff)        
        self.drug_sdf_dir = drug_sdf_dir        
        self._prot2pdb = None
        self._pdb_graph_db = None        
        self._drug2sdf_file = None
        self._drug_sdf_db = None
        self.split_method = split_method
        self.split_frac = split_frac
        self.mmseqs_seq_clus_df = mmseqs_seq_clus_df
        self.seed = seed
        self.onthefly = onthefly

    def _format_pdb_entry(self, _data):
        _coords = _data["coords"]
        entry = {
            "name": _data["name"],
            "seq": _data["seq"],
            "coords": list(zip(_coords["N"], _coords["CA"], _coords["C"], _coords["O"])),
        }        
        if self.emb_dir is not None:
            embed_file = f"{_data['PDB_id']}.{_data['chain']}.pt"
            entry["embed"] = f"{self.emb_dir}/{embed_file}"
        return entry

    @property
    def prot2pdb(self):
        if self._prot2pdb is None:
            self._prot2pdb = {}
            for prot, pdb in self.prot_pdb_id.items():
                _pdb_entry = self.pdb_data[pdb]
                self._prot2pdb[prot] = self._format_pdb_entry(_pdb_entry)
        return self._prot2pdb

    @property
    def pdb_graph_db(self):
        if self._pdb_graph_db is None:
            self._pdb_graph_db = pdb_graph.pdb_to_graphs(self.prot2pdb,
                self.prot_featurize_params)
        return self._pdb_graph_db

    @property
    def drug2sdf_file(self):
        if self._drug2sdf_file is None:            
            drug2sdf_file = {f.stem : str(f) for f in Path(self.drug_sdf_dir).glob('*.sdf')}
            # Convert str keys to int for Davis
            if self.task_name == 'DAVIS' and all([k.isdigit() for k in drug2sdf_file.keys()]):
                drug2sdf_file = {int(k) : v for k, v in drug2sdf_file.items()}
            self._drug2sdf_file = drug2sdf_file
        return self._drug2sdf_file

    @property
    def drug_sdf_db(self):
        if self._drug_sdf_db is None:
            self._drug_sdf_db = mol_graph.sdf_to_graphs(self.drug2sdf_file)
        return self._drug_sdf_db


    def build_data(self, df, onthefly=False):
        records = df.to_dict('records')
        data_list = []
        for entry in records:
            drug = entry['drug']
            prot = entry['protein']
            if onthefly:
                pf = self.prot2pdb[prot]
                df = self.drug2sdf_file[drug]
            else:                
                pf = self.pdb_graph_db[prot]                
                df = self.drug_sdf_db[drug]
            data_list.append({'drug': df, 'protein': pf, 'y': entry['y'],
                'drug_name': drug, 'protein_name': prot})
        if onthefly:
            prot_featurize_fn = partial(
                pdb_graph.featurize_protein_graph,
                **self.prot_featurize_params)            
            drug_featurize_fn = mol_graph.featurize_drug
        else:
            prot_featurize_fn, drug_featurize_fn = None, None
        data = DTA(df=df, data_list=data_list, onthefly=onthefly,
            prot_featurize_fn=prot_featurize_fn, drug_featurize_fn=drug_featurize_fn)
        return data


    def get_split(self, df=None, split_method=None,
            split_frac=None, seed=None, onthefly=None,
            return_df=False):
        df = df or self.df
        split_method = split_method or self.split_method
        split_frac = split_frac or self.split_frac
        seed = seed or self.seed
        onthefly = onthefly or self.onthefly
        if split_method == 'random':
            split_df = create_fold(self.df, seed, split_frac)
        elif split_method == 'drug':
            split_df = create_fold_setting_cold(self.df, seed, split_frac, 'drug')
        elif split_method == 'protein':
            split_df = create_fold_setting_cold(self.df, seed, split_frac, 'protein')
        elif split_method == 'both':
            split_df = create_full_ood_set(self.df, seed, split_frac)
        elif split_method == 'seqid':
            split_df = create_seq_identity_fold(
                self.df, self.mmseqs_seq_clus_df, seed, split_frac)
        else:
            raise ValueError("Unknown split method: {}".format(split_method))
        split_data = {}
        for split, df in split_df.items():
            split_data[split] = self.build_data(df, onthefly=onthefly)
        if return_df:
            return split_data, split_df
        else:
            return split_data


class KIBA(DTATask):
    """
    KIBA drug-target interaction dataset
    """
    def __init__(self,
            data_path='../data/KIBA/kiba_data.tsv',            
            pdb_map='../data/KIBA/kiba_uniprot2pdb.yaml',
            pdb_json='../data/structure/pockets_structure.json',                        
            emb_dir='../data/esm1b',           
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8.,            
            drug_sdf_dir='../data/structure/kiba_mol3d_sdf',
            split_method='random',split_frac =[0.7, 0.1, 0.2],
            mmseqs_seq_cluster_file='../data/KIBA/kiba_cluster_id50_cluster.tsv',
            seed=42, onthefly=False
        ):
        df = pd.read_table(data_path)        
        prot_pdb_id = yaml.safe_load(open(pdb_map, 'r'))
        pdb_data = json.load(open(pdb_json, 'r'))                
        mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_cluster_file, names=['rep', 'seq'])
        super(KIBA, self).__init__(
            task_name='KIBA',
            df=df, 
            prot_pdb_id=prot_pdb_id, pdb_data=pdb_data,
            emb_dir=emb_dir,            
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            split_method=split_method, split_frac=split_frac,
            mmseqs_seq_clus_df=mmseqs_seq_clus_df,
            seed=seed, onthefly=onthefly
            )


class DAVIS(DTATask):
    """
    DAVIS drug-target interaction dataset
    """
    def __init__(self,
            data_path='../data/DAVIS/davis_data.tsv',            
            pdb_map='../data/DAVIS/davis_protein2pdb.yaml',
            pdb_json='../data/structure/pockets_structure.json',                        
            emb_dir='../data/esm1b',           
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8.,            
            drug_sdf_dir='../data/structure/davis_mol3d_sdf',
            split_method='random', split_frac=[0.7, 0.1, 0.2],
            mmseqs_seq_cluster_file='../data/DAVIS/davis_cluster_id50_cluster.tsv',
            seed=42, onthefly=False
        ):
        df = pd.read_table(data_path)        
        prot_pdb_id = yaml.safe_load(open(pdb_map, 'r'))
        pdb_data = json.load(open(pdb_json, 'r'))        
        mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_cluster_file, names=['rep', 'seq'])
        super(DAVIS, self).__init__(
            task_name='DAVIS',
            df=df, 
            prot_pdb_id=prot_pdb_id, pdb_data=pdb_data,
            emb_dir=emb_dir,            
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            split_method=split_method, split_frac=split_frac,
            mmseqs_seq_clus_df=mmseqs_seq_clus_df,
            seed=seed, onthefly=onthefly
            )
