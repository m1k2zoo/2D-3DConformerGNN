# Leveraging 2D Molecular Graph Pretraining for Improved 3D Conformer Generation with Graph Neural Networks 

This repository serves as the official codebase for the research paper titled "Leveraging 2D Molecular Graph Pretraining for Improved 3D Conformer Generation with Graph Neural Networks". This codebase has been significantly influenced by the [GeoMol](https://github.com/PattanaikL/GeoMol) project.

## Enviornment Setup
Please follow the instructions in `create_env.md` to create the conda environment.

## Datasets
To get started, download and extract the GEOM dataset from the original source:

1. Run: `wget https://dataverse.harvard.edu/api/access/datafile/4327252`
2. Run: `tar -xvf 4327252`

## Running Experiments
In this section, we outline the steps to train and produce results for two specific experiments: GeoMol100+MolR100 and GeoMol100+TPSA.

#### Training Procedure
We provide the training scripts along with the path to the trained models.

- **GeoMol100+MolR100**
```bash
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/GeoMol100_MolR100 --n_epochs 250 --dataset qm9 --model_dim 100 --MolR_emb --embed_path 'MolR/saved/sage_100' --embeddings_dim 100
```
Path to model: 'qm9_trained_models\GeoMol100_MolR100\best_model.pt'

- **GeoMol100+TPSA_64**
```bash
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/DeeperGCN_TPSA_64_combine_global --n_epochs 250 --dataset qm9_deepergcn --model_dim 100 --embeddings_dim 64 --utilize_deepergcn --dg_molecular_property TPSA --combine_global
```
Path to model: 'qm9_trained_models\GeoMol100_TPSA_64\best_model.pt'

More training examples are provided in `example_train.md`.

#### Evaluation Procedure
Run the following commands in sequence:

- **GeoMol100+MolR100**
```bash
python generate_confs.py --trained_model_dir qm9_trained_models/GeoMol100_MolR100/ --test_csv data/QM9/test_smiles.csv --dataset qm9 --output_folder GeoMol100_MolR100
python scripts/compare_confs.py --output_folder GeoMol100_MolR100
```

- **GeoMol100+TPSA_64**
```bash
python generate_confs.py --trained_model_dir qm9_trained_models/GeoMol100_TPSA_64/ --test_csv data/QM9/test_smiles.csv --dataset qm9_deepergcn --output_folder GeoMol100_TPSA_64
python scripts/compare_confs.py --output_folder GeoMol100_TPSA_64
```

## Citation
If you find this work useful, please consider citing our paper. The citation information will be provided here soon!