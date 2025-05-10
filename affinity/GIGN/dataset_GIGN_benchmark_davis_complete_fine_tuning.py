# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
import argparse
import random

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')



def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    if not os.path.exists(complex_path):
        print(f"{complex_path} does not exist.")
        return
    # if os.path.exists(save_path):
    #     return
    try: 
        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        x_l, edge_index_l = mol2graph(ligand)
        x_p, edge_index_p = mol2graph(pocket)
        x = torch.cat([x_l, x_p], dim=0)
        edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
        edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        y = torch.FloatTensor([label])
        pos = torch.concat([pos_l, pos_p], dim=0)
        split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
        
        data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

        torch.save(data, save_path)
    # return data
    except Exception as e:
        print(f'cannot process {complex_path}, error: {e}')

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, split_method, split, protein, mutation=None, drug=None, drug_type=None, drug_1_type=None, drug_2_type=None, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False, seed=None, nontruncated_affinity=True):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
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
        self._pre_process()
        
       

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
            

        if self.split_method == 'different_mutation_same_drug' and self.seed:
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, drug_type=self.drug_type, df=self.data_df, seed=self.seed, nontruncated_affinity=self.nontruncated_affinity)
            self.all_graph_list, self.train_graph_list, self.test_graph_list, self.wt_all_graph_list, self.wt_test_graph_list = self.get_split(split_df)
        elif self.split_method == 'different_mutation_same_drug' and not self.seed:
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, drug=self.drug, df=self.data_df, nontruncated_affinity=self.nontruncated_affinity)
            self.all_graph_list, self.train_graph_list, self.test_graph_list, self.wt_all_graph_list, self.wt_test_graph_list = self.get_split(split_df)
        elif self.split_method == 'same_mutation_different_drug' and not self.seed:
            split_df, train_num, test_num = create_fine_tuning_same_mutation_different_drug_split(protein=self.protein, mutation=self.mutation, df=self.data_df, nontruncated_affinity=self.nontruncated_affinity)
            self.all_graph_list, self.train_graph_list, self.test_graph_list, self.wt_all_graph_list, self.wt_test_graph_list = self.get_split(split_df)
        elif self.split_method == 'different_mutation_different_drug':
            split_df, mut_1, mut_2, drug_name_1, drug_name_2 = create_fine_tuning_different_mutation_different_drug_split(protein=self.protein, drug_1_type=self.drug_1_type, drug_2_type=self.drug_2_type, df=self.data_df, seed=self.seed, nontruncated_affinity=self.nontruncated_affinity)
            self.all_graph_list, self.train_graph_list, self.test_graph_list, self.wt_graph_list= self.get_split(split_df)
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))
            

    def __getitem__(self, idx):

        if self.split == 'all':
            return torch.load(self.all_graph_list[idx])
        elif self.split == 'train':
            return torch.load(self.train_graph_list[idx])
        elif self.split == 'test':
            return torch.load(self.test_graph_list[idx])
        elif self.split == 'wt_all':
            return torch.load(self.wt_all_graph_list[idx])
        elif self.split == 'wt_test':
            return torch.load(self.wt_test_graph_list[idx])
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def get_split(self, split_df):

        if split_df is None:
            return [], [], [], []
        
        if self.split_method == 'different_mutation_same_drug' or self.split_method == 'same_mutation_different_drug':
            all_df, train_df, test_df, wt_all_df, wt_test_df = split_df['all'], split_df['train'], split_df['test'], split_df['wt_all'], split_df['wt_test']
            all_graph_list = self.get_graph_list(all_df)
            train_graph_list = self.get_graph_list(train_df)
            test_graph_list = self.get_graph_list(test_df)
            wt_all_graph_list = self.get_graph_list(wt_all_df)
            wt_test_graph_list = self.get_graph_list(wt_test_df)
            return all_graph_list, train_graph_list, test_graph_list, wt_all_graph_list, wt_test_graph_list
        else:
            all_df, train_df, test_df, wt_df = split_df['all'], split_df['train'], split_df['test'], split_df['wt']
            all_graph_list = self.get_graph_list(all_df)
            train_graph_list = self.get_graph_list(train_df)
            test_graph_list = self.get_graph_list(test_df)
            wt_graph_list = self.get_graph_list(wt_df)
            return all_graph_list, train_graph_list, test_graph_list, wt_graph_list
   
    def get_graph_list(self, data_df):
        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        for i, row in data_df.iterrows():
            name = f'{row["protein"]}_{row["drug"]}'
            kd = row['y']
            complex_dir = os.path.join(self.data_dir, name)
            graph_path = os.path.join(complex_dir, f"{self.graph_type}-{name}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{name}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(name)
            pKa_list.append(kd)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        return [graph_path for graph_path in graph_path_list if os.path.exists(graph_path)]


    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        if self.split == 'all':
            return len(self.all_graph_list)
        elif self.split == 'train':
            return len(self.train_graph_list)
        elif self.split == 'test':
            return len(self.test_graph_list)
        elif self.split == 'wt_all':
            return len(self.wt_all_graph_list)
        elif self.split == 'wt_test':
            return len(self.wt_test_graph_list)  
        else:
            raise ValueError(f"Unknown split: {self.split}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--data_df', type=str, default='data/benchmark/davis_data.tsv', help='data of protein and ligand')
    parser.add_argument('--complex_path', type=str, default='data/benchmark/davis_complex_colabfold_diffdock', help='the path of the complexes')
    parser.add_argument('--split_method', choices=['different_mutation_same_drug', 'different_mutation_different_drug'], help='split method')
    parser.add_argument('--split', choices=['train', 'test'], default=None, help='split')
    parser.add_argument('--combination_seed', type=int, default=0, help='random seed for select inhibtor and mutation combination')
    parser.add_argument('--protein', choices=['abl1', 'egfr', 'flt3', 'kit', 'met', 'pik3ca', 'ret'], help='protein name')
    parser.add_argument('--drug_type', choices=['Type I', 'Type II', 'undetermined'], help='drug type', default=None)
    parser.add_argument('--drug_1_type', choices=['Type I', 'Type II', 'undetermined'], help='drug 1 type for different_mutation_different_drug', default=None)
    parser.add_argument('--drug_2_type', choices=['Type I', 'Type II', 'undetermined'], help='drug 2 type for different_mutation_different_drug', default=None)
    
    args = parser.parse_args()

    data_root = args.complex_path
    data_df = args.data_df
    data_df = pd.read_csv(data_df, sep='\t')
    split_method = args.split_method
    split = args.split
    seed = args.seed
    protein = args.protein
    drug_type = args.drug_type
    drug_1_type = args.drug_1_type
    drug_2_type = args.drug_2_type
 
    different_mutation_same_drug_train = GraphDataset(data_root, data_df, split_method='different_mutation_same_drug', split='train', protein=protein, drug_type=drug_type, seed=seed, create=False)
    different_mutation_same_drug_test = GraphDataset(data_root, data_df, split_method='different_mutation_same_drug', split='test', protein=protein, drug_type=drug_type, seed=seed, create=False)

    print (f'split_method: {split_method}')
    print (f"train: {len(different_mutation_same_drug_train)}")
    print (f"test: {len(different_mutation_same_drug_test)}")

    different_mutation_different_drug_train = GraphDataset(data_root, data_df, split_method='different_mutation_different_drug', split='train', protein=protein, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed, create=False)
    different_mutation_different_drug_test = GraphDataset(data_root, data_df, split_method='different_mutation_different_drug', split='test', protein=protein, drug_1_type=drug_1_type, drug_2_type=drug_2_type, seed=seed, create=False)

    print (f'split_method: {split_method}')
    print (f"train: {len(different_mutation_different_drug_train)}")
    print (f"test: {len(different_mutation_different_drug_test)}")

