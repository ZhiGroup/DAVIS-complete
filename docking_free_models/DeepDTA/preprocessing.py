# data processing and training of the DeepDTA paper in pytorch code with your own data
# Author: @ksun63
# Date: 2023-04-14
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
from kdbnet.dta_davis_complete import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold, create_wt_mutation_split


class Dataset(Dataset):
    """
    Here, the input dataset should be a pandas dataframe with the following columns:
    protein, ligands, affinity, where proteins are the protein seqeunces, ligands are the 
    isomeric SMILES representation of the ligand, and affinity is the binding affinity
    """
    def __init__(self, df, seqlen=2000, smilen=200, split=None, split_method=None, seed=None):
        """
        df: pandas dataframe with the columns proteins, ligands, affinity
        seqlen: max length of the protein sequence
        smilen: max length of the ligand SMILES representation
        """
        self.split = split
        self.split_method = split_method
        self.seed = seed
        self.mmseqs_seq_clus_df = pd.read_table('../data/davis_complete/davis_complete_id50_cluster.tsv', names=['rep', 'seq'])
        self.proteins = df['target_sequence'].values
        self.ligands = df['compound_iso_smiles'].values
        self.affinity = df['y'].values
        self.smilelen = smilen
        self.seqlen = seqlen
        self.protein_vocab = set()
        self.ligand_vocab = set()
        for lig in self.ligands:
            for i in lig:
                self.ligand_vocab.update(i)
        for pr in self.proteins:
            for i in pr:
                self.protein_vocab.update(i)

        # having a dummy token to pad the sequences to the max length
        self.protein_vocab.update(['dummy'])
        self.ligand_vocab.update(['dummy'])
        self.protein_vocab = sorted(self.protein_vocab)
        self.ligand_vocab = sorted(self.ligand_vocab)
        self.protein_dict = {x: i for i, x in enumerate(self.protein_vocab)}
        self.ligand_dict = {x: i for i, x in enumerate(self.ligand_vocab)}

        split_frac = [0.7, 0.1, 0.2]

        if self.split_method == 'random':
            self.split_df = create_fold(df, self.seed, split_frac)
        elif self.split_method == 'drug':
            self.split_df = create_fold_setting_cold(df, self.seed, split_frac, 'drug')
        elif self.split_method == 'protein':
            self.split_df = create_fold_setting_cold(df, self.seed, split_frac, 'protein')
        elif self.split_method == 'both':
            self.split_df = create_full_ood_set(df, self.seed, split_frac)
        elif self.split_method == 'seqid':
            self.split_df = create_seq_identity_fold(df, self.mmseqs_seq_clus_df, self.seed, split_frac)
        elif self.split_method == 'wt_mutation':
            self.split_df = create_wt_mutation_split(df, self.seed, [0.9, 0.1])
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))

        if self.split == 'train':
            self.proteins = self.split_df['train']['target_sequence'].values
            self.ligands = self.split_df['train']['compound_iso_smiles'].values
            self.affinity = self.split_df['train']['y'].values
        elif self.split == 'valid':
            self.proteins = self.split_df['valid']['target_sequence'].values
            self.ligands = self.split_df['valid']['compound_iso_smiles'].values
            self.affinity = self.split_df['valid']['y'].values
        elif self.split == 'test':
            self.proteins = self.split_df['test']['target_sequence'].values
            self.ligands = self.split_df['test']['compound_iso_smiles'].values
            self.affinity = self.split_df['test']['y'].values
        elif self.split == 'test_wt':
            if self.split_method == 'wt_mutation':
                self.proteins = None
                self.ligands = None
                self.affinity = None
            else:
                self.proteins = self.split_df['test_wt']['target_sequence'].values
                self.ligands = self.split_df['test_wt']['compound_iso_smiles'].values
                self.affinity = self.split_df['test_wt']['y'].values
        elif self.split == 'test_mutation':
            self.proteins = self.split_df['test_mutation']['target_sequence'].values
            self.ligands = self.split_df['test_mutation']['compound_iso_smiles'].values
            self.affinity = self.split_df['test_mutation']['y'].values
            


    def __len__(self):
        """
        Returns the length of the dataset
        """
        if self.proteins is None:
            return 0
        else:
            return len(self.proteins)

    def __getitem__(self, idx):
        """
        Get the protein, ligand, and affinity of the idx-th sample

        param idx: index of the sample
        """
        pr = self.proteins[idx]
        lig = self.ligands[idx]
        affinity = self.affinity[idx]
        if len(pr) > self.seqlen:
            pr = pr[:self.seqlen]
        if len(lig) > self.smilelen:
            lig = lig[:self.smilelen]
        protein = [self.protein_dict[x] for x in pr] + [self.protein_dict['dummy']] * (self.seqlen - len(pr))
        ligand = [self.ligand_dict[x] for x in lig] + [self.ligand_dict['dummy']] * (self.smilelen - len(lig))

        return torch.tensor(protein), torch.tensor(ligand), torch.tensor(affinity, dtype=torch.float)

def collate_fn(batch):
    """
    Collate function for the DataLoader
    """
    proteins, ligands, affinities = zip(*batch)
    proteins = torch.stack(proteins, dim=0)
    ligands = torch.stack(ligands, dim=0)
    affinities = torch.stack(targets, dim=0)
    return proteins, ligands, affinities


