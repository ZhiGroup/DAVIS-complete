#%%
%reload_ext autoreload
%autoreload 2
from kdbnet.dta_davis_complete import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold
import pandas as pd
from dataset_GIGN_benchmark_davis_complete import GraphDataset, PLIDataLoader


df = pd.read_csv('/data/mwu11/FDA/data/davis_complete/davis_complete.tsv', sep='\t')

split_frac = [0.7, 0.1, 0.2]    
seed = 0
mmseqs_seq_clus_df = pd.read_table('/data/mwu11/FDA/data/davis_complete/davis_complete_id50_cluster.tsv', names=['rep', 'seq'])

split_df_random = create_fold(df, seed, split_frac)
split_df_drug = create_fold_setting_cold(df, seed, split_frac, 'drug')
split_df_protein = create_fold_setting_cold(df, seed, split_frac, 'protein')
split_df_both = create_full_ood_set(df, seed, split_frac)
split_df_seqid = create_seq_identity_fold(df, mmseqs_seq_clus_df, seed, split_frac)


assert len(split_df_random['test']) == len(split_df_random['test_wt']) + len(split_df_random['test_mutation'])
assert len(split_df_drug['test']) == len(split_df_drug['test_wt']) + len(split_df_drug['test_mutation'])
assert len(split_df_protein['test']) == len(split_df_protein['test_wt']) + len(split_df_protein['test_mutation'])
assert len(split_df_both['test']) == len(split_df_both['test_wt']) + len(split_df_both['test_mutation'])
assert len(split_df_seqid['test']) == len(split_df_seqid['test_wt']) + len(split_df_seqid['test_mutation'])



drug_test = GraphDataset(data_root, data_df, split_method='drug', split='test', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=False)




# %%