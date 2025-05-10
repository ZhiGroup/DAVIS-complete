#%%
import torch
import pickle 
esm_embedding = torch.load('data/esm2_3billion_embeddings_davis_colabfold.pt')
# %%



with open('data/cacheNew_torsion_allatoms/limit100_INDEX_davis_colabfold_davis_data_maxLigSize100_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings/heterographs.pkl', 'rb') as f:
    heterographs = pickle.load(f)
# %%
    
complex_name = 'FES_5291'
for graph in heterographs:
    if graph.name  == complex_name:
        esm_embedding_single_protein_graph = graph['receptor'].x[:, 1:]

num_chain = len([ chain_name for chain_name in esm_embedding.keys() if chain_name.split('_')[0] == f'{complex_name.split("_")[0]}'])
esm_embedding_single_protein = []
for i in range(num_chain):
    esm_embedding_single_protein.append(esm_embedding[f'{complex_name.split("_")[0]}_chain_{i}'])
esm_embedding_single_protein = torch.cat(esm_embedding_single_protein, dim=0)

assert  torch.allclose(esm_embedding_single_protein, esm_embedding_single_protein_graph)





# %%
import torch

dict_cuda = {}

for i in range(8):
    dict_cuda[f'cuda{i}'] = torch.device(f'cuda:{i}') 

for i in [7]:
    try:
        x = torch.tensor([1., 2.], device=dict_cuda[f'cuda{i}'])
        print(f'cuda{i} is normal')
    except Exception as e:
        print(f'cuda{i} is not normal')
        print(e)


# %%

import pickle
with open('/data/mwu11/FDA/docking/DiffDock/data/cacheNew_torsion/limit0_INDEX_davis_alphafold_protein_cut_postprocess_davis_data_maxLigSize100_H0_recRad15.0_recMax24_esmEmbeddings/heterograph_ZAP70_9933475.pkl', 'rb') as f:
    heterograph = pickle.load(f)
# %%
