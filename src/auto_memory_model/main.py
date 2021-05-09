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
    parser.add_argument('-data_dir', default=None, help='Data directory', type=str)
    parser.add_argument('-singleton_file', default=None,
                        help='Singleton mentions separately extracted for training.')
    parser.add_argument('-skip_dialog_data', default=False, action="store_true",
                        help='Skip dialog data.')
    parser.add_argument('-base_model_dir', default='../models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument('-model_dir', default=None,
                        help='Model directory', type=str)

    parser.add_argument(
        '-dataset', default='joint_lop', type=str,
        choices=['all', 'joint_lop', 'ontonotes', 'litbank', 'preco', 'wikicoref', 'quizbowl'])
    parser.add_argument(
        '-conll_scorer', type=str, help='Root folder storing model runs',
        default="../resources/lrec2020-coref/reference-coreference-scorers/scorer.pl")

    parser.add_argument('-model_size', default='large', type=str, help='Model size')
    parser.add_argument('-max_segment_len', default=2048, type=int,
                        help='Max segment length of windowed inputs.')
    parser.add_argument('-add_speaker_tokens', default=False, action="store_true",
                        help='Max segment length of windowed inputs.')

    # Mention variables
    parser.add_argument('-max_span_width', default=20, type=int, help='Max span width.')
    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'endpoint'], type=str)
    parser.add_argument('-use_gold_ments', default=False, action="store_true")
    parser.add_argument('-use_topk', default=False, action="store_true", help='Use topk mentions.')
    parser.add_argument('-top_span_ratio', default=0.4, type=float,
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
    # Dataset-specific features
    parser.add_argument('-doc_class', default=None, choices=['dialog', 'genre'],
                        help='What information of document class to use.')

    # Training params
    parser.add_argument('-cross_val_split', default=0, type=int,
                        help='Cross validation split to be used.')
    parser.add_argument('-num_litbank_docs', default=None, type=int,
                        help='Number of litbank training docs.')
    parser.add_argument('-num_ontonotes_docs', default=None, type=int,
                        help='Number of ontonotes training docs.')
    parser.add_argument('-num_preco_docs', default=2500, type=int,
                        help='Number of preco training docs.')
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-num_eval_docs', default=None, type=int,
                        help='Number of evaluation docs.')
    parser.add_argument('-dropout_rate', default=0.3, type=float,
                        help='Dropout rate')
    parser.add_argument('-remove_singletons', default=False, action="store_true",
                        help='Remove singletons from training and eval.')
    parser.add_argument('-label_smoothing_wt', default=0.1, type=float,
                        help='Label Smoothing')
    parser.add_argument('-ment_loss', default='topk', type=str, choices=['all', 'topk'],
                        help='Mention loss computed over topk or all mentions.')
    parser.add_argument('-max_evals',
                        help='Maximum number of evals', default=25, type=int)
    parser.add_argument('-patience',
                        help='Maximum evaluations without improvement', default=5, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-max_gradient_norm',
                        help='Maximum gradient norm', default=1.0, type=float)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=3e-4, type=float)
    parser.add_argument('-fine_tune_lr', help="Fine-tuning learning rate",
                        default=1e-5, type=float)
    parser.add_argument('-eval_per_k_steps', default=None, type=int, help='Evaluate on dev set per k steps')
    parser.add_argument('-update_frequency', default=500, type=int, help='Update freq')
    parser.add_argument('-not_save_model', dest='to_save_model', help="Whether to save model during training or not",
                        default=True, action="store_false")
    parser.add_argument('-eval', dest='eval_model', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'max_segment_len',  # Encoder params
                'ment_emb', 'max_span_width', 'top_span_ratio',  # Mention model
                'mem_type', 'entity_rep', 'mlp_size',  # Memory params
                'dropout_rate', 'seed', 'init_lr', 'max_evals',
                'label_smoothing_wt', 'ment_loss',  # weights & sampling
                'num_ontonotes_docs', 'num_litbank_docs', 'num_preco_docs',
                'sim_func', 'fine_tune_lr', 'doc_class', 'skip_dialog_data',
                'remove_singletons', 'add_speaker_tokens']

    changed_opts = OrderedDict()
    dict_args = vars(args)
    for attr in imp_opts:
        if dict_args[attr] != parser.get_default(attr):
            changed_opts[attr] = dict_args[attr]

    if args.singleton_file is not None and path.exists(args.singleton_file):
        changed_opts['singleton'] = path.basename(args.singleton_file)

    if args.dataset == 'litbank':
        # Cross-validation split is only important for litbank
        changed_opts['cross_val_split'] = args.cross_val_split

    for key, val in vars(args).items():
        if key in changed_opts:
            opt_dict[key] = val

    key_val_pairs = sorted(opt_dict.items())
    str_repr = '_'.join([f'{key}_{val}' for key, val in key_val_pairs])
    model_name = f"longformer_{args.dataset}_" + str_repr
    model_name = model_name.strip('_')

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

    data_dir_dict = {
        'ontonotes': path.join(args.base_data_dir, 'ontonotes/independent_longformer'),
        'preco': path.join(args.base_data_dir, 'preco/independent_longformer'),
        'wikicoref': path.join(args.base_data_dir, 'wikicoref/independent_longformer'),
        'quizbowl': path.join(args.base_data_dir, 'quizbowl/independent_longformer'),
        'litbank': path.join(args.base_data_dir, f'litbank/independent_longformer/{args.cross_val_split}')

    }
    conll_data_dir = {
        'ontonotes': path.join(args.base_data_dir, f'ontonotes/conll'),
        'litbank': path.join(args.base_data_dir, f'litbank/conll/{args.cross_val_split}')
    }

    if args.data_dir is None:
        if args.dataset == 'all':
            args.data_dir_dict = data_dir_dict
            args.conll_data_dir = conll_data_dir
        elif args.dataset == 'joint_lop':
            args.data_dir_dict = {}
            args.conll_data_dir = {}
            for dataset in ['litbank', 'ontonotes', 'preco']:
                args.data_dir_dict[dataset] = data_dir_dict[dataset]
                if dataset in conll_data_dir:
                    args.conll_data_dir[dataset] = conll_data_dir[dataset]
        else:
            if args.dataset != 'litbank':
                args.data_dir_dict = {
                    args.dataset: path.join(args.base_data_dir, f'{args.dataset}/independent_longformer')
                }
            else:
                args.data_dir_dict = {
                    'litbank': path.join(args.base_data_dir, f'litbank/independent_longformer/{args.cross_val_split}')
                }

            args.conll_data_dir = {}
            if args.dataset in conll_data_dir:
                args.conll_data_dir[args.dataset] = conll_data_dir
    else:
        args.data_dir_dict = {args.dataset: args.data_dir}
        args.conll_data_dir = {}

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
