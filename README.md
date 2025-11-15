# DAVIS-complete
A complete and modification-aware version of the [DAVIS dataset](https://www.nature.com/articles/nbt.1990)
by incorporating 4,032 kinase–ligand pairs involving substitutions, insertions, deletions, and phosphorylation events.

The DAVIS-complete benchmark experiment is implemented with Python 3.9.18 and CUDA 11.5 on CentOS Linux 7 (Core), with access to Nvidia A100 (80GB RAM), AMD EPYC 7352 24-Core Processor, and 1TB RAM. 

Run the following to create the environment, DAVIS-complete, which is for running Folding-Docking-Affinity (FDA), DeepDTA, AttentionDTA, GraphDTA, DGraphDTA, and MGraphDTA.
```
conda create --name DAVIS-complete python=3.9
conda activate DAVIS-complete
git clone https://github.com/ZhiGroup/DAVIS-complete
conda install conda-forge::pymol-open-source
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scipy
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch_geometric
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
pip install -e .
```
For Boltz-2 installation,
```
git clone -b train_affinity_module --single-branch https://github.com/AustinApple/boltz.git
cd boltz; pip install -e .
```
## DAVIS-complete Dataset Curation
The script of curating the DAVIS-complete dataset is provided in the `scripts/davis_complete_curation/main.ipynb`.

## Preprocessed Data Download
Download the FDA processed data for replicating the benchmark results from [zenodo](https://zenodo.org/records/15391611) and decompress the files
```
cd DAVIS-complete
mkdir data
cd data
wget https://zenodo.org/records/15391611/files/davis_complete.tar.gz?download=1
tar -xvzf davis_complete.tar.gz?download=1
cd ../
```
Download the Boltz-2 affinity module preprocessed data (the input of affinity module) and decompress the files

```
cd boltz
wget https://zenodo.org/records/17602742/files/boltz2_DAVIS.tar.gz.part-00?download=1
wget https://zenodo.org/records/17604547/files/boltz2_DAVIS.tar.gz.part-01?download=1
wget https://zenodo.org/records/17604708/files/boltz2_DAVIS.tar.gz.part-02?download=1
wget https://zenodo.org/records/17604731/files/boltz2_DAVIS.tar.gz.part-03?download=1
wget https://zenodo.org/records/17604743/files/boltz2_DAVIS.tar.gz.part-04?download=1
wget https://zenodo.org/records/17604753/files/boltz2_DAVIS.tar.gz.part-05?download=1
cat boltz2_DAVIS.tar.gz.part-* > boltz2_DAVIS.tar.gz
tar -xvzf boltz2_DAVIS.tar.gz
mv boltz2_DAVIS DAVIS
cd ../../
```
Download the DGraphDTA processed data for replicating the benchmark results from [zenodo](https://zenodo.org/records/15391611/files/dgraphdta_data.tar.gz?download=1)
```
cd docking_free_models/DGraphDTA/
wget https://zenodo.org/records/15391611/files/dgraphdta_data.tar.gz?download=1
tar -xvzf dgraphdta_data.tar.gz?download=1
mv dgraphdta_data data
cd ../ 
```

## Replicate benchmark results
### Augmented Dataset Prediction
For docking-free based methods, the following command is used to train MGraphDTA, DGraphDTA, GraphDTA, AttentionDTA, and GraphDTA to predict binding affinity under different split_methods (drug_name, drug_structure, protein_modification, protein_name, protein_seqid, protein_modification_drug_name, protein_seqid_drug_structure).
```
cd experiments/docking_free/
python train_script_benchmark.py --split_method drug_name --gpu 0 --model_seeds 0 1 2 3 4 --model_name MGraphDTA
```
For docking-based FDA method, 
```
cd experiments/docking_based/affinity/GIGN
python train_GIGN_benchmark_davis_complete_ensemble.py --split_method drug_name --gpu 0 --seeds 0 1 2 3 4 --job_name davis_complete_drug_name
```
For docking-based Boltz-2 method, 
```
export WORKDIR=/absolute/path/to/your/working/directory
cd boltz/DAVIS/
python scripts/train/train_AffinityModule.py --split_method drug_name --device 0 --max_epochs 100 --batch_size 16 --patience 5 --df_path "$WORKDIR/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv" --mmseqs_cluster_path "$WORKDIR/DAVIS-complete/data/davis_complete/davis_complete_id50_cluster.tsv" --target_dir "$WORKDIR/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input"
```
### Wild-type to modification generalization - Global modification generalization
For docking-free based methods, 
```
cd experiments/docking_free/
python train_script_benchmark.py --split_method wt_mutation --gpu 0 --model_seeds 0 1 2 3 4 --model_name MGraphDTA
```
For docking-based FDA method, 
```
cd experiments/docking_based/affinity/GIGN
python train_GIGN_benchmark_davis_complete_ensemble.py --split_method wt_mutation --gpu 0 --seeds 0 1 2 3 4 --job_name davis_complete_wt_mutation
```
For docking-based Boltz-2 method, 
```
export WORKDIR=/absolute/path/to/your/working/directory
cd boltz/DAVIS/
python scripts/train/train_AffinityModule.py --split_method wt_mutation --device 0 --max_epochs 100 --batch_size 16 --patience 5 --df_path "$WORKDIR/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv" --mmseqs_cluster_path "$WORKDIR/DAVIS-complete/data/davis_complete/davis_complete_id50_cluster.tsv" --target_dir "$WORKDIR/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input"
```

### Same-ligand, different-modifications (Wild-type to modification generalization & Few-shot modification generalization)
For docking-free based methods, 
```
cd experiments/docking_free/
python train_script_fine_tuning.py --split_method different_mutation_same_drug --gpu 0 --model_seeds 0 1 2 3 4 --combination_seed False --epochs 30 --nontruncated_affinity --model_name MGraphDTA 
```
For docking-based FDA method, 
```
cd experiments/docking_based/affinity/GIGN
python train_script_GIGN_benchmark_davis_complete_ensemble_fine_tuning.py --split_method different_mutation_same_drug --gpu 0 --model_seeds 0 1 2 3 4 --combination_seed False --epochs 30 --lr 5e-3 --nontruncated_affinity
```
For docking-based Boltz-2 method,
```
export WORKDIR=/absolute/path/to/your/working/directory
cd boltz/DAVIS/
python scripts/train/finetune_AffinityModule.py --split_method different_mutation_same_drug  --device 0 --max_epochs 10 --df_path "$WORKDIR/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv" --target_dir "$WORKDIR/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input"
```
### Same-modification, different-ligands (Wild-type to modification generalization & Few-shot modification generalization)
For docking-free based methods, 
```
cd docking_free_models
python train_script_fine_tuning.py --split_method same_mutation_different_drug --gpu 0 --model_seeds 0 1 2 3 4 --combination_seed False --lr 1e-4 --epochs 10 --nontruncated_affinity --model_name MGraphDTA 
```
For docking-based FDA method, 
```
cd affinity/GIGN
python train_script_GIGN_benchmark_davis_complete_ensemble_fine_tuning.py --split_method same_mutation_different_drug --gpu 0 --model_seeds 0 1 2 3 4 --combination_seed False --epochs 10 --lr 1e-4 --nontruncated_affinity
```
For docking-based Boltz-2 method,
```
export WORKDIR=/absolute/path/to/your/working/directory
cd boltz/DAVIS/
python scripts/train/finetune_AffinityModule.py --split_method same_mutation_different_drug  --device 0 --max_epochs 10 --df_path "$WORKDIR/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv" --target_dir "$WORKDIR/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input"
```

### Citation
If you find DAVIS-complete useful in your research, please consider citing:
```bibtex
@article{davis2011comprehensive,
  title={Comprehensive analysis of kinase inhibitor selectivity},
  author={Davis, Mindy I and Hunt, Jeremy P and Herrgard, Sanna and Ciceri, Pietro and Wodicka, Lisa M and Pallares, Gabriel and Hocker, Michael and Treiber, Daniel K and Zarrinkar, Patrick P},
  journal={Nature biotechnology},
  volume={29},
  number={11},
  pages={1046--1051},
  year={2011},
  publisher={Nature Publishing Group US New York}
}

@inproceedings{wutowards,
  title={Towards precision protein-ligand affinity prediction benchmark: A Complete and Modification-Aware DAVIS Dataset},
  author={Wu, Ming Hsiu and Xie, Ziqian and Ji, Shuiwang and Zhi, Degui},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```