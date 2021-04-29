import argparse
import os
from os import path
import hashlib
import logging
from collections import OrderedDict

from auto_memory_model.experiment import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_data_dir', default='../data/', help='Root directory of data', type=str)
    parser.add_argument(
        '-data_dir', default=None, help='Data directory. Use this when it is specified', type=str)
    parser.add_argument('-singleton_file', default=None,
                        help='Singleton mentions separately extracted for training.')
    parser.add_argument('-base_model_dir', default='../models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument('-model_dir', default=None,
                        help='Model directory', type=str)

    parser.add_argument(
        '-dataset', default='ontonotes', choices=['litbank', 'ontonotes', 'preco'], type=str)
    parser.add_argument(
        '-conll_scorer', type=str, help='Root folder storing model runs',
        default="../resources/lrec2020-coref/reference-coreference-scorers/scorer.pl")

    parser.add_argument('-model_size', default='large', type=str, help='Model size')
    parser.add_argument('-doc_enc', default='independent', type=str,
                        choices=['independent', 'overlap'], help='Windowing mechanism')
    parser.add_argument('-max_segment_len', default=4096, type=int,
                        help='Max segment length of windowed inputs.')

    # Mention variables
    parser.add_argument('-max_span_width', default=20, type=int, help='Max span width.')
    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'endpoint'], type=str)
    parser.add_argument('-use_gold_ments', default=False, action="store_true")
    parser.add_argument('-top_span_ratio', default=0.3, type=float,
                        help='Ratio of top spans proposed as mentions.')

    # Memory variables
    parser.add_argument('-mem_type', default='unbounded',
                        choices=['learned', 'lru', 'unbounded', 'unbounded_no_ignore'],
                        help="Memory type.")
    parser.add_argument('-mlp_size', default=3000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-cluster_mlp_size', default=3000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers in other MLPs')
    parser.add_argument('-entity_rep', default='wt_avg', type=str,
                        choices=['learned_avg', 'wt_avg', 'max'], help='Entity representation.')
    parser.add_argument('-sim_func', default='hadamard', choices=['hadamard', 'cosine', 'endpoint'],
                        help='Similarity function', type=str)
    parser.add_argument('-emb_size', default=20, type=int,
                        help='Embedding size of features.')
    # Only relevant for bounded memory models
    parser.add_argument('-max_ents', default=20, type=int,
                        help="Number of maximum entities in memory.")
    parser.add_argument('-eval_max_ents', default=None, type=int,
                        help="Number of maximum entities in memory during inference.")

    # Training params
    parser.add_argument('-cross_val_split', default=0, type=int,
                        help='Cross validation split to be used.')
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-num_eval_docs', default=None, type=int,
                        help='Number of evaluation docs.')
    parser.add_argument('-dropout_rate', default=0.3, type=float,
                        help='Dropout rate')
    parser.add_argument('-label_smoothing_wt', default=0.1, type=float,
                        help='Label Smoothing')
    parser.add_argument('-ment_loss', default='topk', type=str, choices=['all', 'topk'],
                        help='Mention loss computed over topk or all mentions.')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=25, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-max_gradient_norm',
                        help='Maximum gradient norm', default=1.0, type=float)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=3e-4, type=float)
    parser.add_argument('-fine_tune_lr', help="Fine-tuning learning rate",
                        default=None, type=float)
    parser.add_argument('-train_with_singletons', help="Train on singletons.",
                        default=False, action="store_true")
    parser.add_argument('-eval_per_k_steps', default=0, type=int, help='Evaluate on dev set per k steps')
    parser.add_argument('-not_save_model', dest='to_save_model', help="Whether to save model during training or not",
                        default=True, action="store_false")
    parser.add_argument('-eval', dest='eval_model', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    if args.dataset in ['litbank', 'preco']:
        args.train_with_singletons = True

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'max_segment_len',  # Encoder params
                'ment_emb', "doc_enc", 'max_span_width', 'top_span_ratio',  # Mention model
                'mem_type', 'entity_rep', 'mlp_size',  # Memory params
                'dropout_rate', 'seed', 'init_lr', 'max_epochs',
                'label_smoothing_wt', 'ment_loss',  # weights & sampling
                'num_train_docs', 'train_with_singletons',  'dataset',  # Dataset params
                ]

    if args.singleton_file is not None and path.exists(args.singleton_file):
        imp_opts.append('singleton_file')

    if args.fine_tune_lr is not None:
        imp_opts.append('fine_tune_lr')

    if args.sim_func is not parser.get_default('sim_func'):
        imp_opts.append('sim_func')

    # Adding conditional important options
    if args.mem_type in ['learned', 'lru']:
        # Number of max entities only matters for bounded memory models
        imp_opts.append('max_ents')
    else:
        args.max_ents = None

    if args.dataset == 'litbank':
        # Cross-validation split is only important for litbank
        imp_opts.append('cross_val_split')

    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = f"longformer_{args.dataset}_" + str(hash_idx)

    if args.eval_model:
        args.max_training_segments = None

    if args.model_dir is None:
        model_dir = path.join(args.base_model_dir, model_name)
        args.model_dir = model_dir
        best_model_dir = path.join(model_dir, 'best_models')
        args.best_model_dir = best_model_dir
        if not path.exists(model_dir):
            os.makedirs(model_dir)
        if not path.exists(best_model_dir):
            os.makedirs(best_model_dir)
    else:
        best_model_dir = path.join(args.model_dir, 'best_models')
        if not path.exists(best_model_dir):
            best_model_dir = args.model_dir
        args.best_model_dir = best_model_dir

    print("Model directory:", args.model_dir)

    if args.data_dir is None:
        if args.dataset == 'litbank':
            args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}/{args.cross_val_split}')
            args.conll_data_dir = path.join(args.base_data_dir, f'{args.dataset}/conll/{args.cross_val_split}')
        elif args.dataset == 'ontonotes':
            if args.train_with_singletons:
                enc_str = "_singletons"
            else:
                enc_str = ""
            args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}{enc_str}')
            args.conll_data_dir = path.join(args.base_data_dir, f'{args.dataset}/conll')
        else:
            args.conll_data_dir = None
    else:
        base_dir = path.dirname(args.data_dir.rstrip("/"))
        if args.dataset == 'litbank':
            args.data_dir = path.join(args.data_dir, f'{args.cross_val_split}')
            args.conll_data_dir = path.join(base_dir, f'conll/{args.cross_val_split}')
        elif args.dataset == 'ontonotes':
            args.conll_data_dir = path.join(base_dir, "conll")
        else:
            args.conll_data_dir = None

    print(args.data_dir)

    # Log directory for Tensorflow Summary
    log_dir = path.join(args.model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    config_file = path.join(args.model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    Experiment(**vars(args))


if __name__ == "__main__":
    main()
