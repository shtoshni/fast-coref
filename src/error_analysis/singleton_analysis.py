import argparse
import os
import logging
import json
import numpy as np
from coref_utils.metrics import CorefEvaluator
from coref_utils.utils import get_mention_to_cluster, filter_clusters

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger()


def process_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument("log_file", help="Log file", type=str)

    args = parser.parse_args()
    return args


def singleton_analysis(data):
    gold_singletons = 0
    pred_singletons = 0

    # singleton_evaluator = CorefEvaluator()
    non_singleton_evaluator = CorefEvaluator()

    gold_cluster_lens = []
    pred_cluster_lens = []

    overlap_sing = 0
    total_sing = 0
    pred_sing = 0

    for instance in data:
        # Singleton performance
        gold_clusters = set(
            [tuple(cluster[0]) for cluster in instance["clusters"] if len(cluster) == 1]
        )
        pred_clusters = set(
            [
                tuple(cluster[0])
                for cluster in instance["predicted_clusters"]
                if len(cluster) == 1
            ]
        )

        total_sing += len(gold_clusters)
        pred_sing += len(pred_clusters)
        overlap_sing += len(gold_clusters.intersection(pred_clusters))

        gold_singletons += len(gold_clusters)
        pred_singletons += len(pred_clusters)

        # predicted_clusters, mention_to_predicted = get_mention_to_cluster(pred_clusters, threshold=1)
        # gold_clusters, mention_to_gold = get_mention_to_cluster(gold_clusters, threshold=1)
        # singleton_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        # Non-singleton performance
        gold_clusters = filter_clusters(instance["clusters"], threshold=2)
        pred_clusters = filter_clusters(instance["predicted_clusters"], threshold=2)

        gold_cluster_lens.extend([len(cluster) for cluster in instance["clusters"]])
        pred_cluster_lens.extend(
            [len(cluster) for cluster in instance["predicted_clusters"]]
        )

        # gold_clusters = filter_clusters(gold_clusters, threshold=1)
        # pred_clusters = filter_clusters(pred_clusters, threshold=1)

        mention_to_predicted = get_mention_to_cluster(pred_clusters)
        mention_to_gold = get_mention_to_cluster(gold_clusters)
        non_singleton_evaluator.update(
            pred_clusters, gold_clusters, mention_to_predicted, mention_to_gold
        )

    logger.info(
        "\nGT singletons: %d, Pred singletons: %d\n"
        % (gold_singletons, pred_singletons)
    )
    recall_sing = overlap_sing / total_sing
    pred_sing = overlap_sing / pred_sing
    f_sing = 2 * recall_sing * pred_sing / (recall_sing + pred_sing)
    logger.info(
        f"\nSingletons - Recall: {recall_sing * 100: .1f}, Pred: {pred_sing * 100: .1f}, "
        f"F1: {f_sing * 100: .1f}\n"
    )
    logger.info(
        f"\nNon-singleton cluster lengths, Gold: {np.mean(gold_cluster_lens):.2f}, "
        f"Pred: {np.mean(pred_cluster_lens):.2f}\n"
    )

    for evaluator, evaluator_str in zip([non_singleton_evaluator], ["Non-singleton"]):
        perf_str = ""
        indv_metrics_list = ["MUC", "BCub", "CEAFE"]
        for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator.evaluators):
            # perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)
            perf_str += "{} - {}".format(indv_metric, indv_evaluator.get_prf_str())

        fscore = evaluator.get_f1() * 100
        perf_str += "{:.1f} ".format(fscore)
        perf_str = perf_str.strip(", ")
        logger.info("\n%s\n%s\n" % (evaluator_str, perf_str))


def main():
    args = process_args()
    data = []
    with open(args.log_file) as f:
        for line in f:
            data.append(json.loads(line))
    singleton_analysis(data)


if __name__ == "__main__":
    main()
