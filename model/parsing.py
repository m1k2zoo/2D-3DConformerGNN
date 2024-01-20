from argparse import ArgumentParser, Namespace


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--split_path', type=str)
    parser.add_argument('--trained_local_model', type=str)
    parser.add_argument('--restart_dir', type=str)
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--seed', type=int, default=0)

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4) # TODO: solve issue with >0 not working

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--verbose', action='store_true', default=False)

    # Model arguments
    # parser.add_argument('--model_dim', type=int, default=25)
    parser.add_argument('--model_dim', type=int, default=50)
    parser.add_argument('--random_vec_dim', type=int, default=10)
    parser.add_argument('--random_vec_std', type=float, default=1)
    parser.add_argument('--random_alpha', action='store_true', default=False)
    parser.add_argument('--n_true_confs', type=int, default=10)
    parser.add_argument('--n_model_confs', type=int, default=10)

    parser.add_argument('--gnn1_depth', type=int, default=3)
    parser.add_argument('--gnn1_n_layers', type=int, default=2)
    parser.add_argument('--gnn2_depth', type=int, default=3)
    parser.add_argument('--gnn2_n_layers', type=int, default=2)
    parser.add_argument('--encoder_n_head', type=int, default=2)
    parser.add_argument('--coord_pred_n_layers', type=int, default=2)
    parser.add_argument('--d_mlp_n_layers', type=int, default=1)
    parser.add_argument('--h_mol_mlp_n_layers', type=int, default=1)
    parser.add_argument('--alpha_mlp_n_layers', type=int, default=2)
    parser.add_argument('--c_mlp_n_layers', type=int, default=1)

    parser.add_argument('--global_transformer', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default='ot_emd')
    parser.add_argument('--teacher_force', action='store_true', default=False)
    parser.add_argument('--separate_opts', action='store_true', default=False)

    parser.add_argument('--no_h_mol', action='store_true', default=False) # Ablation with no global information
    parser.add_argument('--MolR_emb', action='store_true', default=False) # pretrained embedding using reactions
    parser.add_argument('--MolR_node_emb', action='store_true', default=False) # use reactions node embedding
    parser.add_argument('--embed_path', type=str, default='MolR/saved/sage_50')
    parser.add_argument('--embeddings_dim', type=int, default=50)
    parser.add_argument('--combine_global', action='store_true', default=False) # combine MolR global embeddings with h_mol from GeoMol

    parser.add_argument('--utilize_fingerprints', action='store_true', default=False) # Use fingerprints instead of h_mol
    parser.add_argument('--augment_fingerprints', action='store_true', default=False) # Use fingerprints in addition to h_mol
    
    parser.add_argument('--utilize_deepergcn', action='store_true', default=False) # Use deepergcn instead of h_mol
    parser.add_argument('--dg_molecular_property', type=str, default='TPSA')
    
def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args


def set_hyperparams(args):
    """
    Converts ArgumentParser args to hyperparam dictionary.

    :param args: Namespace containing the args.
    :return: A dictionary containing the args as hyperparams.
    """

    hyperparams = {'model_dim': args.model_dim,
                   'random_vec_dim': args.random_vec_dim,
                   'random_vec_std': args.random_vec_std,
                   'global_transformer': args.global_transformer,
                   'n_true_confs': args.n_true_confs,
                   'n_model_confs': args.n_model_confs,
                   'gnn1': {'depth': args.gnn1_depth,
                            'n_layers': args.gnn1_n_layers},
                   'gnn2': {'depth': args.gnn2_depth,
                            'n_layers': args.gnn2_n_layers},
                   'encoder': {'n_head': args.encoder_n_head},
                   'coord_pred': {'n_layers': args.coord_pred_n_layers},
                   'd_mlp': {'n_layers': args.d_mlp_n_layers},
                   'h_mol_mlp': {'n_layers': args.h_mol_mlp_n_layers},
                   'alpha_mlp': {'n_layers': args.alpha_mlp_n_layers},
                   'c_mlp': {'n_layers': args.c_mlp_n_layers},
                   'loss_type': args.loss_type,
                   'teacher_force': args.teacher_force,
                   'random_alpha': args.random_alpha,
                   'no_h_mol': args.no_h_mol,
                   'MolR_emb': args.MolR_emb,
                   'MolR_node_emb': args.MolR_node_emb,
                    'embed_path': args.embed_path,
                    'embeddings_dim': args.embeddings_dim,
                    'combine_global': args.combine_global,
                    'utilize_fingerprints': args.utilize_fingerprints,
                    'augment_fingerprints': args.augment_fingerprints,
                    'utilize_deepergcn': args.utilize_deepergcn,
                    'dg_molecular_property': args.dg_molecular_property}

    return hyperparams
