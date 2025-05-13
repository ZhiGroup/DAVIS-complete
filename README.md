# DAVIS-complete
A complete, modification-aware version of the [DAVIS dataset](https://www.nature.com/articles/nbt.1990)
by incorporating 4,032 kinase–ligand pairs involving substitutions, insertions, deletions, and phosphorylation events.

The DAVIS-complete benchmark experiment is implemented with Python 3.9.18 and CUDA 11.5 on CentOS Linux 7 (Core), with access to Nvidia A100 (80GB RAM), AMD EPYC 7352 24-Core Processor, and 1TB RAM. 

Run the following to create the environment, DAVIS-complete.

```
conda create --name DAVIS-complete python=3.9
conda activate FDA
conda install conda-forge::pymol-open-source
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scipy
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch_geometric
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
```
## Datasets
Download the FDA processed data for replicating the benchmark results from [zenodo](https://zenodo.org/records/15391611) and decompress the files

```
git clone https://github.com/ZhiGroup/DAVIS-complete
cd DAVIS-complete
mkdir data
cd data
wget https://zenodo.org/records/15391611/files/davis_complete.tar.gz?download=1
tar -xvzf davis_complete.tar.gz?download=1
cd ../
```
Download the DGraphDTA processed data for replicating the benchmark results from [zenodo](https://zenodo.org/records/15391611/files/dgraphdta_data.tar.gz?download=1)
```
cd docking_free_models/DGraphDTA/
wget https://zenodo.org/records/15391611/files/dgraphdta_data.tar.gz?download=1
tar -xvzf dgraphdta_data.tar.gz?download=1
mv dgraphdta_data data
cd ../ 
```

 
## Replicate results
### Augmented Dataset Prediction
For docking-free based methods, the following command is used to train MGraphDTA, DGraphDTA, GraphDTA, AttentionDTA, and GraphDTA to predict binding affinity under different split_methods (both, drug, protein, and seqid).
```
cd docking_free_models
python train_script_benchmark.py --split_method both --gpu 3 --model_seeds 0 1 2 3 4 --model_name MGraphDTA
```
For docking-based FDA method, 
```
cd affinity/GIGN
python train_GIGN_benchmark_davis_complete_ensemble.py --split_method both --gpu 3 --seeds 0 1 2 3 4
```
### Wild-type to modification generalization - Global modification generalization
For docking-free based methods, 
```
cd docking_free_models
python train_script_benchmark.py --split_method wt_mutation --gpu 3 --model_seeds 0 1 2 3 4 --model_name MGraphDTA
```
For docking-based FDA method, 
```
cd affinity/GIGN
python train_GIGN_benchmark_davis_complete_ensemble.py --split_method wt_mutation --gpu 3 --seeds 0 1 2 3 4
```
### Same-ligand, different-modifications (Wild-type to modification generalization & Few-shot modification generalization)
For docking-free based methods, 
```
cd docking_free_models
python train_script_fine_tuning.py --split_method different_mutation_same_drug --gpu 3 --model_seeds 0 1 2 3 4 --combination_seed False --epochs 30 --nontruncated_affinity --model_name MGraphDTA 
```
For docking-based FDA method, 
```
cd affinity/GIGN
python train_script_GIGN_benchmark_davis_complete_ensemble_fine_tuning.py --split_method different_mutation_same_drug --gpu 3 --model_seeds 0 1 2 3 4 --combination_seed False --epochs 30 --lr 5e-3 --nontruncated_affinity
```
### Same-modification, different-ligands (Wild-type to modification generalization & Few-shot modification generalization)
For docking-free based methods, 
```
cd docking_free_models
python train_script_fine_tuning.py --split_method same_mutation_different_drug --gpu 3 --model_seeds 0 1 2 3 4 --combination_seed False --lr 1e-4 --epochs 10 --nontruncated_affinity --model_name MGraphDTA 
```
For docking-based FDA method, 
```
cd affinity/GIGN
python train_script_GIGN_benchmark_davis_complete_ensemble_fine_tuning.py --split_method same_mutation_different_drug --gpu 3 --model_seeds 0 1 2 3 4 --combination_seed False --epochs 10 --lr 1e-4 --nontruncated_affinity
```


