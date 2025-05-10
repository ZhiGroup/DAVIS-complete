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
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
from itertools import product


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

   
    # @property
    # def raw_file_names(self):
    #     return ['davis_complete.csv']

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

        # if self.pre_filter is not None:
        #     all_list = [all for all in all_list if self.pre_filter(all)]
        #     train_list = [train for train in train_list if self.pre_filter(train)]
        #     test_list = [test for test in test_list if self.pre_filter(test)]
        #     wt_list = [wt for wt in wt_list if self.pre_filter(wt)]

        # if self.pre_transform is not None:
        #     all_list = [self.pre_transform(all) for all in all_list]
        #     train_list = [self.pre_transform(train) for train in train_list]
        #     test_list = [self.pre_transform(test) for test in test_list]
        #     wt_list = [self.pre_transform(wt) for wt in wt_list]
    

        print('Graph construction done. Saving to file.')

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
   

