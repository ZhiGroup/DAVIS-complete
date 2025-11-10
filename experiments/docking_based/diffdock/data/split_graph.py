#%%
import pickle
import os
# Specify the file path where the list is stored
file_path = "/data/mwu11/FDA/docking/DiffDock/data/cacheNew_torsion/limit0_INDEX_kiba_colabfold_protein_kiba_data_maxLigSize100_H0_recRad15.0_recMax24_esmEmbeddings/heterographs.pkl"

# Load the list from the file
with open(file_path, "rb") as file:
    heterographs = pickle.load(file)

#%%
parent_dir = "/data/mwu11/FDA/docking/DiffDock/data/cacheNew_torsion/limit0_INDEX_kiba_colabfold_protein_kiba_data_maxLigSize100_H0_recRad15.0_recMax24_esmEmbeddings/"
for heterograph in heterographs:
    name = heterograph.name
    save_path = os.path.join(parent_dir, f'{name}.pkl')
    with open(save_path, "wb") as file:
        pickle.dump(heterograph, file)

# %%
