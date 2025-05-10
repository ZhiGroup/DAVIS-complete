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
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
from itertools import product


class Dataset(Dataset):
    """
    Here, the input dataset should be a pandas dataframe with the following columns:
    protein, ligands, affinity, where proteins are the protein seqeunces, ligands are the 
    isomeric SMILES representation of the ligand, and affinity is the binding affinity
    """
    def __init__(self, df, split_method, split, protein, mutation=None, seqlen=2000, smilen=200, drug=None, drug_type=None, drug_1_type=None, drug_2_type=None, seed=None, nontruncated_affinity=True, transform=None, pre_transform=None, pre_filter=None):
        """
        df: pandas dataframe with the columns proteins, ligands, affinity
        seqlen: max length of the protein sequence
        smilen: max length of the ligand SMILES representation
        """
        self.seed = seed
        self.proteins = df['target_sequence'].values
        self.ligands = df['compound_iso_smiles'].values
        self.affinity = df['y'].values

        self.split_method = split_method
        self.split = split

        self.protein = protein
        self.mutation = mutation
        self.drug = drug
        self.drug_type = drug_type
        self.drug_1_type = drug_1_type
        self.drug_2_type = drug_2_type
        self.seed = seed
        self.nontruncated_affinity = nontruncated_affinity
        
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

        if self.split_method == 'different_mutation_same_drug' and self.seed:
            self.split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, drug_type=self.drug_type, df=df, seed=self.seed, nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'different_mutation_same_drug' and not self.seed:
            self.split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, drug=self.drug, df=df, nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'same_mutation_different_drug':
            self.split_df, train_num, test_num = create_fine_tuning_same_mutation_different_drug_split(protein=self.protein, mutation=self.mutation, df=df, nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'different_mutation_different_drug':
            self.split_df, mut_1, mut_2, drug_name_1, drug_name_2 = create_fine_tuning_different_mutation_different_drug_split(protein=self.protein, drug_1_type=self.drug_1_type, drug_2_type=self.drug_2_type, df=df, seed=self.seed, nontruncated_affinity=self.nontruncated_affinity)
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))

        if self.split == 'all':
            self.proteins = self.split_df['all']['target_sequence'].values
            self.ligands = self.split_df['all']['compound_iso_smiles'].values
            self.affinity = self.split_df['all']['y'].values
        elif self.split == 'train':
            self.proteins = self.split_df['train']['target_sequence'].values
            self.ligands = self.split_df['train']['compound_iso_smiles'].values
            self.affinity = self.split_df['train']['y'].values
        elif self.split == 'test':
            self.proteins = self.split_df['test']['target_sequence'].values
            self.ligands = self.split_df['test']['compound_iso_smiles'].values
            self.affinity = self.split_df['test']['y'].values
        elif self.split == 'wt_all':
            self.proteins = self.split_df['wt_all']['target_sequence'].values
            self.ligands = self.split_df['wt_all']['compound_iso_smiles'].values
            self.affinity = self.split_df['wt_all']['y'].values
        elif self.split == 'wt_test':
            self.proteins = self.split_df['wt_test']['target_sequence'].values
            self.ligands = self.split_df['wt_test']['compound_iso_smiles'].values
            self.affinity = self.split_df['wt_test']['y'].values
        else:
            raise ValueError("Unknown split: {}".format(self.split))
        
        
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
    affinities = torch.stack(affinities, dim=0)
    return proteins, ligands, affinities


