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

[Leveraging 2D molecular graph pretraining for improved 3D conformer generation with graph neural networks](https://doi.org/10.1016/j.compchemeng.2024.108622)

```
@article{ALHAMOUD2024108622,
title = {Leveraging 2D molecular graph pretraining for improved 3D conformer generation with graph neural networks},
journal = {Computers & Chemical Engineering},
volume = {183},
pages = {108622},
year = {2024},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2024.108622},
url = {https://www.sciencedirect.com/science/article/pii/S0098135424000401},
author = {Kumail Alhamoud and Yasir Ghunaim and Abdulelah S. Alshehri and Guohao Li and Bernard Ghanem and Fengqi You},
keywords = {Graph neural networks, Conformation generation, Pretraining molecular graph embeddings, Drug design, 3D molecular modeling},
abstract = {Predicting stable 3D molecular conformations from 2D molecular graphs is a challenging and resource-intensive task, yet it is critical for various applications, particularly drug design. Density functional theory (DFT) calculations set the standard for molecular conformation generation, yet they are computationally intensive. Deep learning offers more computationally efficient approaches, but struggles to match DFT accuracy, particularly on complex drug-like structures. Additionally, the steep computational demands of assembling 3D molecular datasets constrain the broader adoption of deep learning. This work aims to utilize the abundant 2D molecular graph datasets for pretraining a machine learning model, a step that involves initially training the model on a different task with a wealth of data before fine-tuning it for the target task of 3D conformation generation. We build on GeoMol, an end-to-end graph neural network (GNN) method for predicting atomic 3D structures and torsion angles. We examine the limitations of the GeoMol method and introduce new baselines to enhance molecular graph embeddings. Our computational results show that 2D molecular graph pretraining enhances the quality of generated 3D conformers, yielding a 7.7 % average improvement over state-of-the-art sequential methods. These advancements not only facilitate superior 3D conformation generation but also emphasize the potential of leveraging pretrained graph embeddings to boost performance in 3D chemical tasks with GNNs.}
}
```
