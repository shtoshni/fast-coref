import argparse
import os
from os import path
import logging

from mention_model.experiment import Experiment
from mention_model.utils import get_mention_model_name


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_data_dir', default='../data', help='Root directory of data', type=str)
    parser.add_argument(
        '-data_dir', default=None, help='Data directory. Use this when it is specified', type=str)
    parser.add_argument(
        '-dataset', default='litbank', choices=['litbank', 'ontonotes'], type=str)
    parser.add_argument('-base_model_dir',
                        default='../models', help='Root folder storing model runs', type=str)
    parser.add_argument('-model_dir', default=None, help='Model directory', type=str)
    parser.add_argument('-singleton_file', default=None,
                        help='Singleton mentions separately extracted for training.')
    parser.add_argument('-model_size', default='large', type=str,
                        help='Model size')
    parser.add_argument('-doc_enc', default='independent', type=str,
                        choices=['independent', 'overlap'], help='BERT model type')
    parser.add_argument('-max_segment_len', default=2048, type=int,
                        help='Max segment length of BERT segments.')
    parser.add_argument('-max_training_segments', default=1, type=int,
                        help='Maximum number of BERT segments in a document.')
    parser.add_argument('-top_span_ratio', default=0.3, type=float,
                        help='Ratio of top spans proposed as mentions.')

    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'max', 'endpoint'],
                        type=str)
    parser.add_argument('-max_span_width',
                        help='Max span width', default=20, type=int)
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers in other MLPs')
    parser.add_argument('-mlp_size', default=3000, type=int,
                        help='MLP size used in the model')

    parser.add_argument('-cross_val_split', default=0, type=int,
                        help='Cross validation split to be used.')
    parser.add_argument('--batch_size', '-bsize',
                        help='Batch size', default=1, type=int)
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-num_eval_docs', default=None, type=int,
                        help='Number of evaluation docs.')
    parser.add_argument('-dropout_rate', default=0.3, type=float,
                        help='Dropout rate')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=10, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=5e-4, type=float)
    parser.add_argument('-fine_tune_lr', help="Fine-tuning learning rate",
                        default=None, type=float)
    parser.add_argument('-train_with_singletons', default=False, action="store_true",
                        help='Whether to use singletons during training or not.')
    parser.add_argument('-checkpoint', help="Use checkpoint",
                        default=False, action="store_true")
    parser.add_argument('-eval', dest='eval_model', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    if args.model_dir is None:
        model_name = get_mention_model_name(args)
        print(model_name)

        model_dir = path.join(args.base_model_dir, model_name)
        args.model_dir = model_dir
        best_model_dir = path.join(model_dir, 'best_models')
        args.best_model_dir = best_model_dir
        if not path.exists(model_dir):
            os.makedirs(model_dir)
        if not path.exists(best_model_dir):
            os.makedirs(best_model_dir)
    else:
        args.best_model_dir = args.model_dir

    if args.data_dir is None:
        if args.dataset == 'litbank':
            args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}/{args.cross_val_split}')
        else:
            args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}')

    Experiment(args, **vars(args))


if __name__ == "__main__":
    # import torch
    # torch.multiprocessing.set_start_method('spawn')
    main()
