import os
from os import path
import logging
from omegaconf import OmegaConf
import hydra
import hashlib
import json
import wandb

from experiment import Experiment

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model_name(config):
    masked_copy = OmegaConf.masked_copy(
        config, ["datasets", "model", "trainer", "optimizer"]
    )
    encoded = json.dumps(OmegaConf.to_container(masked_copy), sort_keys=True).encode()
    # encoded['seed']=
    hash_obj = hashlib.md5()
    hash_obj.update(encoded)
    hash_obj.update(f"seed: {config.seed}".encode())

    model_hash = str(hash_obj.hexdigest())
    if len(config.datasets) > 1:
        dataset_name = "joint"
    else:
        dataset_name = list(config.datasets.keys())[0]
        if dataset_name == "litbank":
            cross_val_split = config.datasets[dataset_name].cross_val_split
            dataset_name += f"_cv_{cross_val_split}"

    model_name = f"{dataset_name}_{model_hash}"
    return model_name


def main_train(config):
    if config.paths.model_name is None:
        model_name = get_model_name(config)
    else:
        model_name = config.paths.model_name

    config.paths.model_dir = path.join(
        config.paths.base_model_dir, config.paths.model_name_prefix + model_name
    )
    config.paths.best_model_dir = path.join(config.paths.model_dir, "best")

    for model_dir in [config.paths.model_dir, config.paths.best_model_dir]:
        if not path.exists(model_dir):
            os.makedirs(model_dir)

    if config.paths.model_path is None:
        config.paths.model_path = path.abspath(
            path.join(config.paths.model_dir, config.paths.model_filename)
        )
        config.paths.best_model_path = path.abspath(
            path.join(config.paths.best_model_dir, config.paths.model_filename)
        )

    if config.paths.best_model_path is None and (config.paths.model_path is not None):
        config.paths.best_model_path = config.paths.model_path

    # Dump config file
    config_file = path.join(config.paths.model_dir, "config.json")
    with open(config_file, "w") as f:
        f.write(json.dumps(OmegaConf.to_container(config), indent=4, sort_keys=True))

    return model_name


def main_eval(config):
    if config.paths.model_dir is None:
        raise ValueError

    best_model_dir = path.join(config.paths.model_dir, "best")
    if path.exists(best_model_dir):
        config.paths.best_model_dir = best_model_dir
    else:
        config.paths.best_model_dir = config.paths.model_dir

    config.paths.best_model_path = path.abspath(
        path.join(config.paths.best_model_dir, config.paths.model_filename)
    )


@hydra.main(config_path="conf", config_name="config")
def main(config):
    if config.train:
        model_name = main_train(config)
    else:
        main_eval(config)
        model_name = path.basename(path.normpath(config.paths.model_dir))
        # Strip prefix
        if model_name.startswith(config.paths.model_name_prefix):
            model_name = model_name[len(config.paths.model_name_prefix) :]

    if config.use_wandb:
        # Wandb Initialization
        try:
            wandb.init(
                id=model_name,
                project="Coreference",
                config=dict(config),
                resume=True,
            )
        except:
            # Turn off wandb
            config.use_wandb = False

    logger.info(f"Model name: {model_name}")
    Experiment(config)


if __name__ == "__main__":
    import sys

    sys.argv.append(f"hydra.run.dir={path.dirname(path.realpath(__file__))}")
    sys.argv.append("hydra/job_logging=none")
    main()
