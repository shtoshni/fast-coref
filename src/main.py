import argparse
import os
from os import path
import torch
import logging
from omegaconf import OmegaConf
import hydra
import hashlib
import json

from experiment import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg):
	masked_copy = OmegaConf.masked_copy(cfg, ['dataset', 'model', 'trainer', 'optimizer'])
	encoded = json.dumps(OmegaConf.to_container(masked_copy), sort_keys=True).encode()
	hashlib.md5().update(encoded)

	model_name = str(hashlib.md5().hexdigest())
	cfg.paths.model_dir = path.join(cfg.paths.base_model_dir, cfg.paths.model_name_prefix + model_name)
	cfg.paths.best_model_dir = path.join(cfg.paths.model_dir, 'best')

	if cfg.paths.model_path is None:
		cfg.paths.model_path = path.join(cfg.paths.model_dir, cfg.paths.model_filename)
	if cfg.paths.best_model_path is None:
		cfg.paths.best_model_path = path.join(cfg.paths.best_model_dir, cfg.paths.model_filename)

	# print(cfg.datasets['litbank'])
	# print(dict(cfg))
	print(cfg.paths.conll_scorer)
	Experiment(cfg)


if __name__ == "__main__":
	import sys

	sys.argv.append(f'hydra.run.dir={os.path.expanduser("~")}')
	hydra_main()
