# Examples of training GeoMol with pretrained MolR models
## GeoMol100 + MolR100:
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/GeoMol100_MolR100 --n_epochs 250 --dataset qm9 --model_dim 100 --MolR_emb --embed_path 'MolR/saved/sage_100' --embeddings_dim 100

## GeoMol100 + MolR1024:
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/GeoMol100_MolR1024 --n_epochs 250 --dataset qm9 --model_dim 100 --MolR_emb --embed_path 'MolR/saved/sage_1024' --embeddings_dim 1024


# Examples of training GeoMol with pretrained DeeperGCN models on various molecular properties 
## GeoMol100 + TPSA
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/DeeperGCN_TPSA_64_combine_global --n_epochs 250 --dataset qm9_deepergcn --model_dim 100 --embeddings_dim 64 --utilize_deepergcn --dg_molecular_property TPSA --combine_global

## GeoMol100 + logP
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/DeeperGCN_logP_64_combine_global --n_epochs 250 --dataset qm9_deepergcn --model_dim 100 --embeddings_dim 64 --utilize_deepergcn --dg_molecular_property logP --combine_global

## GeoMol100 + qed
python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./qm9_trained_models/DeeperGCN_qed_64_combine_global --n_epochs 250 --dataset qm9_deepergcn --model_dim 100 --embeddings_dim 64 --utilize_deepergcn --dg_molecular_property qed --combine_global