import pandas as pd
import numpy as np
import os, glob
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from tqdm import tqdm
from kdbnet.dta_davis_complete import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold, create_wt_mutation_split
import torch

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot, max_seq_len=1000):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x  


class GNNDataset(InMemoryDataset):
    def __init__(self, df, root='/tmp', split=None, transform=None, pre_transform=None, split_method=None, seed=None):
        self.df = df 
        self.split_method = split_method
        self.mmseqs_seq_clus_df = pd.read_table('../data/davis_complete/davis_complete_id50_cluster.tsv', names=['rep', 'seq'])
        self.seed = seed
        super().__init__(root, transform, pre_transform)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
        elif split == 'test_wt':
            self.data, self.slices = torch.load(self.processed_paths[3])
        elif split == 'test_mutation':
            self.data, self.slices = torch.load(self.processed_paths[4])
        else:
            raise ValueError("Unknown split: {}".format(split))
        
   

    # @property
    # def raw_file_names(self):
    #     return ['davis_complete.csv']

    @property
    def processed_file_names(self):
        return [f'processed_data_{self.split_method}_train_{self.seed}.pt',\
                f'processed_data_{self.split_method}_valid_{self.seed}.pt',\
                f'processed_data_{self.split_method}_test_{self.seed}.pt',\
                f'processed_data_{self.split_method}_test_wt_{self.seed}.pt',\
                f'processed_data_{self.split_method}_test_mutation_{self.seed}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: list of graphs in PyTorch-Geometric format
    def process_data(self, df, smile_graph):
        
        data_list = []

        if df is None:
            return data_list

        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['y']


            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smi]
            target = seq_cat(sequence)

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([label]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        return data_list
        
    def process(self):
        
        # df = pd.read_csv(self.raw_paths[0])
        df = self.df 
        graph_dict = {}
        all_smiles = df['compound_iso_smiles'].unique()
        for smiles in tqdm(all_smiles):
            graph_dict[smiles] = smiles_to_graph(smiles)


        split_frac = [0.7, 0.1, 0.2]

        if self.split_method == 'random':
            split_df = create_fold(df, self.seed, split_frac)
        elif self.split_method == 'drug':
            split_df = create_fold_setting_cold(df, self.seed, split_frac, 'drug')
        elif self.split_method == 'protein':
            split_df = create_fold_setting_cold(df, self.seed, split_frac, 'protein')
        elif self.split_method == 'both':
            split_df = create_full_ood_set(df, self.seed, split_frac)
        elif self.split_method == 'seqid':
            split_df = create_seq_identity_fold(df, self.mmseqs_seq_clus_df, self.seed, split_frac)
        elif self.split_method == 'wt_mutation':
            split_df = create_wt_mutation_split(df, self.seed, [0.9, 0.1])
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))
        
        train_list = self.process_data(split_df['train'], graph_dict)
        valid_list = self.process_data(split_df['valid'], graph_dict)
        test_list = self.process_data(split_df['test'], graph_dict)
        test_wt_list = self.process_data(split_df['test_wt'], graph_dict)
        test_mutation_list = self.process_data(split_df['test_mutation'], graph_dict)
        
        
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(train_list)
        # save preprocessed train data:
        torch.save((data, slices), self.processed_paths[0])
        data, slices = self.collate(valid_list)
        # save preprocessed valid data:
        torch.save((data, slices), self.processed_paths[1])

        data, slices = self.collate(test_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[2])
        
        if not self.split_method == 'wt_mutation':
            data, slices = self.collate(test_wt_list)
            # save preprocessed test data:
            torch.save((data, slices), self.processed_paths[3])

        data, slices = self.collate(test_mutation_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[4])


if __name__ == "__main__":
    root = 'data/davis_complete'
    GNNDataset(root=root, split_method='random')
    GNNDataset(root=root, split_method='protein')
    GNNDataset(root=root, split_method='drug')
    GNNDataset(root=root, split_method='both')
    GNNDataset(root=root, split_method='seqid')

