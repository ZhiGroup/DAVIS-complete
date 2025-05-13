import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from utils import *
from kdbnet.dta_davis_complete import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold, create_wt_mutation_split
import pickle
from tqdm import tqdm
import pickle

# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# print(res_weight_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index


# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    # k = float(len(pro_res_table))
    # pwm_mat = np.log2(ppm_mat / (1.0 / k))
    # pssm_mat = pwm_mat
    # print(pssm_mat)
    return pssm_mat


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    # print('target_feature')
    # print(pssm.shape)
    # print(other_feature.shape)

    # print(other_feature.shape)
    # return other_feature
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence)
    return feature


# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index


# to judge whether the required files exist
def valid_target(key, dataset):
    contact_dir = 'data/' + dataset + '/pconsc4'
    aln_dir = 'data/' + dataset + '/aln'
    contact_file = os.path.join(contact_dir, key + '.npy')
    aln_file = os.path.join(aln_dir, key + '.aln')
    # print(contact_file, aln_file)
    if os.path.exists(contact_file) and os.path.exists(aln_file):
        return True
    else:
        return False


def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')




class GNNDataset(InMemoryDataset):

    def __init__(self, df, root, split=None, transform=None, pre_transform=None, pre_filter=None, split_method=None, seed=None):
        self.df = df
        self.split_method = split_method
        self.mmseqs_seq_clus_df = pd.read_table('../data/davis_complete/davis_complete_id50_cluster.tsv', names=['rep', 'seq'])
        self.seed = seed
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

        
   

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


    def process_data(self, df, smile_graph, target_graph):

        if df is None:
            return [], [] 

        data_list_ligand = []
        data_list_protein = []

        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            protein = row['protein']
            sequence = row['target_sequence']
            label = row['y']

            c_size, features, edge_index = smile_graph[smi]
            target_size, target_features, target_edge_index = target_graph[protein]
            
            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([label]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([label]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
           

            data_list_ligand.append(GCNData_mol)
            data_list_protein.append(GCNData_pro)

        assert len(data_list_ligand) == len(data_list_protein)
        return data_list_ligand, data_list_protein

    def process(self):
        df = self.df
        # df = pd.read_csv(self.raw_paths[0])

        # get ligand graph
        smiles = df['compound_iso_smiles'].unique()
        
        if not os.path.exists('smile_graph.pkl'):
            smile_graph = dict()
            for smile in smiles:
                g = smile_to_graph(smile)
                smile_graph[smile] = g
            with open('smile_graph.pkl', 'wb') as file:
                pickle.dump(smile_graph, file)
        else:
            with open('smile_graph.pkl', 'rb') as file:
                smile_graph = pickle.load(file)
        

        with open('DGraphDTA/data/davis_complete/dict_proteinname_seq.pkl', "rb") as file:
            dict_proteinname_seq = pickle.load(file)

        msa_path = os.path.join('DGraphDTA', 'data', 'davis_complete', 'aln')
        contac_path = os.path.join('DGraphDTA', 'data', 'davis_complete', 'pconsc4')
        
        if not os.path.exists('target_graph.pkl'):
            target_graph = dict()
            for proteinname in dict_proteinname_seq.keys():
                g = target_to_graph(proteinname, dict_proteinname_seq[proteinname], contac_path, msa_path)
                target_graph[proteinname] = g
            with open('target_graph.pkl', 'wb') as file:
                pickle.dump(target_graph, file)
        else:
            with open('target_graph.pkl', 'rb') as file:
                target_graph = pickle.load(file)
        
        
        
        split_frac=[0.7, 0.1, 0.2]

        if self.split_method == 'random':
            split_df = create_fold(df, self.seed, split_frac)
        elif self.split_method == 'drug':
            split_df = create_fold_setting_cold(df, self.seed, split_frac, 'drug_name')
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
        
        self.train_list_ligand,  self.train_list_protein = self.process_data(split_df['train'], smile_graph, target_graph)
        self.valid_list_ligand, self.valid_list_protein = self.process_data(split_df['valid'], smile_graph, target_graph)
        self.test_list_ligand, self.test_list_protein = self.process_data(split_df['test'], smile_graph, target_graph)
        self.test_wt_list_ligand, self.test_wt_list_protein = self.process_data(split_df['test_wt'], smile_graph, target_graph)
        self.test_mutation_list_ligand, self.test_mutation_list_protein = self.process_data(split_df['test_mutation'], smile_graph, target_graph)

       

    def __len__(self):
        if self.split == 'train':
            return len(self.train_list_ligand)
        elif self.split == 'valid':
            return len(self.valid_list_ligand)
        elif self.split == 'test':
            return len(self.test_list_ligand)
        elif self.split == 'test_wt':
            return len(self.test_wt_list_ligand)
        elif self.split == 'test_mutation':
            return len(self.test_mutation_list_ligand)
        else:
            raise ValueError("Unknown split: {}".format(self.split))

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_list_protein[idx], self.train_list_ligand[idx]
        elif self.split == 'valid':
            return self.valid_list_protein[idx], self.valid_list_ligand[idx]
        elif self.split == 'test':
            return self.test_list_protein[idx], self.test_list_ligand[idx]
        elif self.split == 'test_wt':
            return self.test_wt_list_protein[idx], self.test_wt_list_ligand[idx]
        elif self.split == 'test_mutation':
            return self.test_mutation_list_protein[idx], self.test_mutation_list_ligand[idx]
        else:
            raise ValueError("Unknown split: {}".format(self.split))

    

if __name__ == "__main__":
    root = 'data/davis_complete'
    GNNDataset(root, split='train', split_method='protein', seed=0)
    # GNNDataset(root=root, split_method='protein')
    # GNNDataset(root=root, split_method='drug')
    # GNNDataset(root=root, split_method='both')
    # GNNDataset(root=root, split_method='seqid')