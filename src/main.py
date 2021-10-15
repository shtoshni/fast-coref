import os
from os import path
import logging
from omegaconf import OmegaConf
import hydra
import hashlib
import json

from experiment import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
	masked_copy = OmegaConf.masked_copy(cfg, ['datasets', 'model', 'trainer', 'optimizer'])
	print(masked_copy)
	encoded = json.dumps(OmegaConf.to_container(masked_copy), sort_keys=True).encode()
	hash_obj = hashlib.md5()
	hash_obj.update(encoded)

	model_name = str(hash_obj.hexdigest())
	cfg.paths.model_dir = path.join(cfg.paths.base_model_dir, cfg.paths.model_name_prefix + model_name)
	cfg.paths.best_model_dir = path.join(cfg.paths.model_dir, 'best')

	for model_dir in [cfg.paths.model_dir, cfg.paths.best_model_dir]:
		if not path.exists(model_dir):
			os.makedirs(model_dir)

	if cfg.paths.model_path is None:
		cfg.paths.model_path = path.join(cfg.paths.model_dir, cfg.paths.model_filename)
	if cfg.paths.best_model_path is None:
		cfg.paths.best_model_path = path.join(cfg.paths.best_model_dir, cfg.paths.model_filename)

	print(OmegaConf.to_yaml(cfg))
	Experiment(cfg)


if __name__ == "__main__":
	import sys
	sys.argv.append(f'hydra.run.dir={os.path.expanduser("~")}')
	main()
