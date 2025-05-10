import pandas as pd
import numpy as np
import os, glob
import json,pickle
from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch

from torch_geometric.data import InMemoryDataset, DataLoader
from tqdm import tqdm
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
from itertools import product

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

    def __init__(self, df, root, split_method, split, protein, mutation=None, drug=None, drug_type=None, drug_1_type=None, drug_2_type=None, seed=None, nontruncated_affinity=True, transform=None, pre_transform=None, pre_filter=None):
        self.df = df
        self.protein = protein
        self.mutation = mutation
        self.drug = drug
        self.drug_type = drug_type
        self.drug_1_type = drug_1_type
        self.drug_2_type = drug_2_type
        self.split_method = split_method
        self.split = split
        self.seed = seed
        self.nontruncated_affinity = nontruncated_affinity
        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'all':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
        elif split == 'wt_all':
            self.data, self.slices = torch.load(self.processed_paths[3])
        elif split == 'wt_test':
            self.data, self.slices = torch.load(self.processed_paths[4])
        else:
            raise ValueError("Unknown split: {}".format(split))
        
   

    @property
    def raw_file_names(self):
        return ['davis_complete.csv']

    @property
    def processed_file_names(self):
        if self.split_method == 'different_mutation_same_drug' or self.split_method == 'same_mutation_different_drug':
            return [f'processed_data_{self.split_method}_all_protein{self.protein}_mutation{self.mutation}_drug{self.drug}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_train_protein{self.protein}_mutation{self.mutation}_drug{self.drug}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_test_protein{self.protein}_mutation{self.mutation}_drug{self.drug}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_wt_all_protein{self.protein}_mutation{self.mutation}_drug{self.drug}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_wt_test_protein{self.protein}_mutation{self.mutation}_drug{self.drug}_cobminationseed_{self.seed}.pt']
        elif self.split_method == 'different_mutation_different_drug':
            return [f'processed_data_{self.split_method}_all_protein{self.protein}_drug1{self.drug_1_type}_drug2{self.drug_2_type}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_train_protein{self.protein}_drug1{self.drug_1_type}_drug2{self.drug_2_type}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_test_protein{self.protein}_drug1{self.drug_1_type}_drug2{self.drug_2_type}_cobminationseed_{self.seed}.pt',\
                    f'processed_data_{self.split_method}_wt_protein{self.protein}_drug1{self.drug_1_type}_drug2{self.drug_2_type}_cobminationseed_{self.seed}.pt']
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))

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


        if self.split_method == 'different_mutation_same_drug' and self.seed:
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, drug_type=self.drug_type, df=df, seed=self.seed, nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'different_mutation_same_drug' and not self.seed:
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, drug=self.drug, df=df, nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'same_mutation_different_drug':
            split_df, train_num, test_num = create_fine_tuning_same_mutation_different_drug_split(protein=self.protein, mutation=self.mutation, df=df, nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'different_mutation_different_drug':
            split_df, mut_1, mut_2, drug_name_1, drug_name_2 = create_fine_tuning_different_mutation_different_drug_split(protein=self.protein, drug_1_type=self.drug_1_type, drug_2_type=self.drug_2_type, df=df, seed=self.seed, nontruncated_affinity=self.nontruncated_affinity)
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))
        
        all_list = self.process_data(split_df['all'], graph_dict)
        train_list = self.process_data(split_df['train'], graph_dict)
        test_list = self.process_data(split_df['test'], graph_dict)
        wt_all_list = self.process_data(split_df['wt_all'], graph_dict)
        wt_test_list = self.process_data(split_df['wt_test'], graph_dict)
        
        data, slices = self.collate(all_list)
        # save preprocessed all data:
        torch.save((data, slices), self.processed_paths[0])

        data, slices = self.collate(train_list)
        # save preprocessed train data:
        torch.save((data, slices), self.processed_paths[1])
        
        data, slices = self.collate(test_list)
        # save preprocessed valid data:
        torch.save((data, slices), self.processed_paths[2])

        data, slices = self.collate(wt_all_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[3])

        data, slices = self.collate(wt_test_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[4])


if __name__ == "__main__":
    protein = ['abl1', 'egfr', 'flt3', 'kit', 'met', 'pik3ca', 'ret']
    ligand = list(pd.read_csv('/data/mwu11/FDA/data/davis_complete/davis_inhibitor_binding_mode.csv')['Compound'])
    combinations = list(product(protein, ligand))

    root = 'data/davis_complete_finetuning'
    combination_seed = False
    drug_type, drug_1_type, drug_2_type = None, None, None
    data_df = pd.read_csv('data/davis_complete_finetuning/raw/davis_complete.csv')
    for protein, drug_name in combinations:

        split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=protein, drug=drug_name, df=data_df, seed=combination_seed, nontruncated_affinity=True)
        if not split_df:
            continue

        print(protein, drug_name, train_num, test_num)
        GNNDataset(root=root, split_method='different_mutation_same_drug', split='train', protein=protein, drug=drug_name if not combination_seed else None, drug_type=drug_type, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=combination_seed)

