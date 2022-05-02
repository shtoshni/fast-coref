import os
import logging
import time
import json
import torch
from os import path
from collections import OrderedDict, Counter

from coref_utils.metrics import CorefEvaluator
from coref_utils.conll import evaluate_conll
from coref_utils.utils import get_mention_to_cluster, is_aligned, filter_clusters

from model.utils import action_sequences_to_clusters
from model.entity_ranking_model import EntityRankingModel

from omegaconf import DictConfig
from typing import Dict
from torch import Tensor

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


def full_coref_evaluation(
    config: DictConfig,
    model: EntityRankingModel,
    data_iter_map: Dict,
    dataset: str,
    split="dev",
    final_eval=False,
    conll_data_dir: Dict = None,
) -> Dict:
    """Function to evaluate full coreference chains.

    Args:
            config: Experiment configuration
            model: Coreference model
            data_iter_map: Data iterator
            dataset: Name of the coreference dataset
            split: Partition of the dataset - train/dev/test
            final_eval: Whether this is a periodic evaluation or final evaluation
                    For final evaluation, official CoNLL scores can be calculated if possible.
            conll_data_dir:  Data directory dictionary which maps datasets to their gold CoNLL files.

    Returns:
            dict: Dictionary with results for all the metrics.
    """

    # Measure time
    inference_time = 0.0

    dataset_config: DictConfig = config.datasets[dataset]
    cluster_threshold: int = dataset_config["cluster_threshold"]
    logger.info(f"Dataset: {dataset}, Cluster Threshold: {cluster_threshold}")

    log_dir = path.join(config.paths.model_dir, dataset)
    if not path.exists(log_dir):
        os.makedirs(log_dir)
    gold_ment_str = ""
    if config.model.mention_params.use_gold_ments:
        gold_ment_str = "_gold"

    log_file = path.join(log_dir, split + gold_ment_str + ".log.jsonl")

    # Reset the peak memory to compute max memory stat for inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with open(log_file, "w") as f:
        # Capture the auxiliary action accuracy
        corr_actions, total_actions = 0.0, 0.0
        oracle_evaluator, evaluator = CorefEvaluator(), CorefEvaluator()
        coref_predictions, subtoken_maps = {}, {}

        logger.info(f"Evaluating on {len(data_iter_map[split][dataset])} examples")
        for example in data_iter_map[split][dataset]:
            start_time = time.time()
            pred_mentions, mention_scores, gt_actions, pred_actions = model(example)

            # Process predicted clusters
            raw_predicted_clusters = action_sequences_to_clusters(
                pred_actions, pred_mentions
            )
            predicted_clusters = filter_clusters(
                raw_predicted_clusters, threshold=cluster_threshold
            )
            mention_to_predicted = get_mention_to_cluster(predicted_clusters)

            gold_clusters = filter_clusters(
                example["clusters"], threshold=cluster_threshold
            )
            mention_to_gold = get_mention_to_cluster(gold_clusters)
            evaluator.update(
                predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
            )

            elapsed_time = time.time() - start_time
            inference_time += elapsed_time

            coref_predictions[example["doc_key"]] = predicted_clusters
            if "orig_subtoken_map" in example:
                subtoken_maps[example["doc_key"]] = example["orig_subtoken_map"]
            else:
                subtoken_maps[example["doc_key"]] = example["subtoken_map"]

            total_actions += len(pred_actions)

            # Oracle clustering - Best performance possible given the predicted mentions
            oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
            oracle_clusters = filter_clusters(
                oracle_clusters, threshold=cluster_threshold
            )
            mention_to_oracle = get_mention_to_cluster(oracle_clusters)
            oracle_evaluator.update(
                oracle_clusters, gold_clusters, mention_to_oracle, mention_to_gold
            )

            log_example = dict(example)
            log_example["pred_mentions"] = pred_mentions
            log_example["mention_scores"] = mention_scores
            if cluster_threshold != 1:
                # For cluster threshold 1, raw and processed clusters are one and the same
                log_example["raw_predicted_clusters"] = raw_predicted_clusters

            log_example["gt_actions"] = gt_actions
            log_example["pred_actions"] = pred_actions
            log_example["predicted_clusters"] = predicted_clusters

            del log_example["tensorized_sent"]
            for key in list(log_example.keys()):
                if isinstance(log_example[key], Tensor):
                    del log_example[key]

            f.write(json.dumps(log_example) + "\n")

        result_dict: Dict = OrderedDict()
        perf_str: str = ""
        # Print individual metrics
        for indv_metric, indv_evaluator in zip(config.metrics, evaluator.evaluators):
            perf_str += (
                ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)
            )
            result_dict[indv_metric] = OrderedDict()
            result_dict[indv_metric]["recall"] = round(
                indv_evaluator.get_recall() * 100, 1
            )
            result_dict[indv_metric]["precision"] = round(
                indv_evaluator.get_precision() * 100, 1
            )
            result_dict[indv_metric]["fscore"] = round(indv_evaluator.get_f1() * 100, 1)

        result_dict["fscore"] = round(evaluator.get_f1() * 100, 1)
        logger.info("F-score: %.1f %s" % (result_dict["fscore"], perf_str))

        try:
            # Check if the dataset has CoNLL annotations to begin with
            if not dataset_config.get("has_conll", False):
                return result_dict

            # (1) Only use CoNLL evaluator script for final evaluation
            # (2) CoNLL score only makes sense when the evaluation is using the canonical cluster threshold
            # (3) Check if the scorer and CoNLL annotation directory exist
            is_canonical = (
                dataset_config.cluster_threshold
                == dataset_config.canonical_cluster_threshold
            )
            try:
                path_exists_bool = path.exists(
                    config.paths.conll_scorer
                ) and path.exists(conll_data_dir[dataset])
            except TypeError:
                # This exception occurs when NoneType is passed along
                path_exists_bool = False

            if final_eval and is_canonical and path_exists_bool:
                logger.info("\n\nUsing CoNLL scorer")
                gold_path = path.join(conll_data_dir[dataset], f"{split}.conll")
                prediction_file = path.join(log_dir, f"{split}.conll")

                print(path.abspath(gold_path))
                print(path.abspath(prediction_file))
                print(config.paths.conll_scorer)

                conll_results = evaluate_conll(
                    config.paths.conll_scorer,
                    gold_path,
                    coref_predictions,
                    subtoken_maps,
                    prediction_file,
                )

                for indv_metric in config.metrics:
                    result_dict[indv_metric]["recall"] = round(
                        conll_results[indv_metric.lower()]["r"], 1
                    )
                    result_dict[indv_metric]["precision"] = round(
                        conll_results[indv_metric.lower()]["p"], 1
                    )
                    result_dict[indv_metric]["fscore"] = round(
                        conll_results[indv_metric.lower()]["f"], 1
                    )

                average_f1 = sum(
                    results["f"] for results in conll_results.values()
                ) / len(conll_results)
                result_dict["fscore"] = round(average_f1, 1)

                logger.info(
                    "(CoNLL) F-score : %.1f, MUC: %.1f, Bcub: %.1f, CEAFE: %.1f"
                    % (
                        average_f1,
                        conll_results["muc"]["f"],
                        conll_results["bcub"]["f"],
                        conll_results["ceafe"]["f"],
                    )
                )
                logger.info("Prediction file: %s" % path.abspath(prediction_file))
        except AttributeError:
            pass

        logger.info("Oracle F-score: %.3f" % oracle_evaluator.get_prf()[2])
        logger.info(path.abspath(log_file))
        logger.handlers[0].flush()

    logger.info("Inference time: %.2f" % inference_time)
    max_mem = (
        (torch.cuda.max_memory_allocated() / (1024**3))
        if torch.cuda.is_available()
        else 0.0
    )
    logger.info("Max inference memory: %.1f GB" % max_mem)

    return result_dict


def targeted_coref_evaluation(
    config: DictConfig,
    model: EntityRankingModel,
    data_iter_map: Dict,
    dataset: str,
    split="test",
) -> Dict:
    """Function to perform targeted coreference evaluation for datasets such as GAP.

    Datasets such as GAP and WSC only provide annotation for specific coreference pairs.
    Due to this we need to use a separate evaluation function.
    The vanilla F-score is used as the evaluation metric.
    """

    # Set up logging paths
    log_dir = path.join(config.paths.model_dir, dataset)
    if not path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = path.join(log_dir, split + ".log.jsonl")

    with open(log_file, "w") as f:
        logger.info(f"Evaluating on {len(data_iter_map[split][dataset])} examples")
        # Counter for keeping track of the key stats
        counter: Dict = Counter()
        for document in data_iter_map[split][dataset]:
            pred_mentions, mention_scores, gt_actions, pred_actions = model(document)

            log_example = dict(document)
            del log_example["tensorized_sent"]
            for key in list(log_example.keys()):
                if isinstance(log_example[key], Tensor):
                    del log_example[key]

            predicted_clusters = action_sequences_to_clusters(
                pred_actions, pred_mentions
            )
            predicted_clusters = filter_clusters(predicted_clusters, threshold=1)
            mention_to_predicted = get_mention_to_cluster(predicted_clusters)

            pron_span = tuple(document["pronoun_span"])
            a_pred, b_pred = False, False  # Default prediction is assumed to be False

            if pron_span in mention_to_predicted:
                pron_cluster = mention_to_predicted[pron_span]
                for span in pron_cluster:
                    a_aligned = is_aligned(span, tuple(document["a_span"]))
                    b_aligned = is_aligned(span, tuple(document["b_span"]))

                    if a_aligned:
                        a_pred = True
                    if b_aligned:
                        b_pred = True

            if dataset == "wsc":
                span_not_found = False
                for span in [
                    document["a_span"],
                    document["b_span"],
                    document["pronoun_span"],
                ]:
                    if tuple(span) not in mention_to_predicted:
                        span_not_found = True
                        break

                if span_not_found:
                    counter["span_not_found"] += 1

                corr = (a_pred == document["a_label"]) and (
                    b_pred == document["b_label"]
                )
                log_example["correct"] = corr
                counter["corr"] += (a_pred == document["a_label"]) and (
                    b_pred == document["b_label"]
                )
                counter["total"] += 1

            elif dataset == "gap":
                for gt, pred in zip(
                    [document["a_label"], document["b_label"]], [a_pred, b_pred]
                ):
                    if gt and pred:
                        counter["true_positive"] += 1
                    elif gt and (not pred):
                        counter["false_negative"] += 1
                    elif (not gt) and (not pred):
                        counter["true_negative"] += 1
                    else:
                        counter["false_positive"] += 1

            else:
                raise ValueError(
                    f"Dataset {dataset} evaluation is currently not supported"
                )

            log_example["a_pred"] = a_pred
            log_example["b_pred"] = b_pred
            log_example["predicted_clusters"] = predicted_clusters
            f.write(json.dumps(log_example) + "\n")

    logger.info(path.abspath(log_file))

    result_dict = {"fscore": 0.0}
    if dataset == "wsc":
        result_dict = {"fscore": (counter["corr"] * 100) / counter["total"]}
        logger.info("Accuracy: %.1f" % result_dict["fscore"])
        logger.info(
            "Span not found: %.1f%%"
            % ((counter["span_not_found"] * 100) / counter["total"])
        )
    elif dataset == "gap":
        prec = counter["true_positive"] / (
            counter["true_positive"] + counter["false_positive"]
        )
        recall = counter["true_positive"] / (
            counter["true_positive"] + counter["false_negative"]
        )

        result_dict["prec"], result_dict["recall"] = prec * 100, recall * 100
        if prec and recall:
            result_dict = {"fscore": (2 * prec * recall * 100) / (prec + recall)}

        logger.info("F-score: %.1f" % result_dict["fscore"])

    return result_dict


def coref_evaluation(
    config: DictConfig,
    model: EntityRankingModel,
    data_iter_map: Dict,
    dataset: str,
    split="dev",
    final_eval=False,
    conll_data_dir: Dict = None,
) -> Dict:
    """Evaluation function which calls the dataset-appropriate coreference evaluation function."""

    dataset_config = config.datasets[dataset]
    if dataset_config.get("targeted_eval", False):
        return targeted_coref_evaluation(
            config, model, data_iter_map, dataset, split=split
        )
    else:
        return full_coref_evaluation(
            config,
            model,
            data_iter_map,
            dataset,
            split=split,
            final_eval=final_eval,
            conll_data_dir=conll_data_dir,
        )
