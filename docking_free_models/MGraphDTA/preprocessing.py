import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
from kdbnet.dta_davis_complete import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold, create_wt_mutation_split



'''
Note that training and test datasets are the same as GraphDTA
Please see: https://github.com/thinng/GraphDTA
'''

VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 


class GNNDataset(InMemoryDataset):

    def __init__(self, df, root, split=None, transform=None, pre_transform=None, pre_filter=None, split_method=None, seed=None):
        self.df = df
        self.split_method = split_method
        self.mmseqs_seq_clus_df = pd.read_table('../data/davis_complete/davis_complete_id50_cluster.tsv', names=['rep', 'seq'])
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)

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


    def process_data(self, df, graph_dict):

        data_list = []

        if df is None:
            return data_list
        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['y']

            x, edge_index, edge_attr = graph_dict[smi]

            # caution
            x = (x - x.min()) / (x.max() - x.min())

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len- len(target)))
            else:
                target = target[:target_len]

            # Get Labels
            try:
                data = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.FloatTensor([label]),
                    target=torch.LongTensor([target])
                )
            except:
                    print("unable to process: ", smi)

            data_list.append(data)

        return data_list

    def process(self):
        df = self.df
        # df = pd.read_csv(self.raw_paths[0])
        smiles = df['compound_iso_smiles'].unique()
        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            graph_dict[smile] = g

        
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
        
        train_list = self.process_data(split_df['train'], graph_dict)
        valid_list = self.process_data(split_df['valid'], graph_dict)
        test_list = self.process_data(split_df['test'], graph_dict)
        test_wt_list = self.process_data(split_df['test_wt'], graph_dict)
        test_mutation_list = self.process_data(split_df['test_mutation'], graph_dict)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            valid_list = [valid for valid in train_list if self.pre_filter(valid)]
            test_list = [test for test in test_list if self.pre_filter(test)]
            test_wt_list = [test_wt for test_wt in test_wt_list if self.pre_filter(test_wt)]
            test_mutation_list = [test_mutation for test_mutation in test_mutation_list if self.pre_filter(test_mutation)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            valid_list = [self.pre_transform(valid) for valid in valid_list]
            test_list = [self.pre_transform(test) for test in test_list]
            test_wt_list = [self.pre_transform(test_wt) for test_wt in test_wt_list]
            test_mutation_list = [self.pre_transform(test_mutation) for test_mutation in test_mutation_list]

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

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])

        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr

if __name__ == "__main__":
    root = 'data/davis_complete'
    GNNDataset(root=root, split_method='random')
    GNNDataset(root=root, split_method='protein')
    GNNDataset(root=root, split_method='drug')
    GNNDataset(root=root, split_method='both')
    GNNDataset(root=root, split_method='seqid')

