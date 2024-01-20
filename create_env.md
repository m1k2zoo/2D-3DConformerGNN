# Create env
conda create -n 3d_conf python=3.9
conda activate 3d_conf

# Install pytorch and CUDA 10.2
conda install pytorch==1.11.0 torchvision cudatoolkit=10.2 -c pytorch

# Install torch-scatter torch-sparse torch-cluster
pip install torch-scatter==2.0.9 torch-cluster torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu102.html


# Test dependencies installation
python -c "from torch_scatter import scatter_max"
python -c "from torch_sparse import coalesce"
python -c "from torch_cluster import *"

# Install torch-geometric
pip install torch-geometric==1.7.2

# Test torch_geometric installation
python -c "import torch_geometric"

# Install dependencies
conda install tensorboard jupyter rdkit pot yaml pyyaml psutil seaborn -c conda-forge -c rdkit -c anaconda
pip install dgl-cu102 pysmiles

# Test installation 
python generate_confs.py --trained_model_dir qm9_trained_models/GeoMol100/ --test_csv data/QM9/test_smiles.csv --dataset qm9 --output_folder GeoMol100
