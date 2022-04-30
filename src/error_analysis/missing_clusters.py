import argparse
import os
import logging
import json
import numpy as np
from coref_utils.metrics import CorefEvaluator
from coref_utils.utils import get_mention_to_cluster


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
    max_length = 0
    max_doc_id = ""
    max_cluster = []

    for instance in data:

        gold_clusters, gold_mentions_to_cluster = get_mention_to_cluster(
            instance["clusters"]
        )
        pred_clusters, pred_mentions_to_cluster = get_mention_to_cluster(
            instance["predicted_clusters"]
        )

        for cluster in gold_clusters:
            all_mention_unseen = True
            for mention in cluster:
                if mention in pred_mentions_to_cluster:
                    all_mention_unseen = False
                    break

            if all_mention_unseen:
                if len(cluster) > max_length:
                    max_length = len(cluster)
                    max_doc_id = instance["doc_key"]
                    max_cluster = cluster

    print(max_doc_id)
    print(max_length, max_cluster)


def reverse_analysis(data):
    max_length = 0
    max_doc_id = ""
    max_cluster = []

    for instance in data:

        gold_clusters, gold_mentions_to_cluster = get_mention_to_cluster(
            instance["clusters"]
        )
        pred_clusters, pred_mentions_to_cluster = get_mention_to_cluster(
            instance["predicted_clusters"]
        )

        for cluster in pred_clusters:
            all_mention_unseen = True
            for mention in cluster:
                if mention in gold_mentions_to_cluster:
                    all_mention_unseen = False
                    break

            if all_mention_unseen:
                if len(cluster) > max_length:
                    max_length = len(cluster)
                    max_doc_id = instance["doc_key"]
                    max_cluster = cluster

    print(max_doc_id)
    print(max_length, max_cluster)


def main():
    args = process_args()
    data = []
    with open(args.log_file) as f:
        for line in f:
            data.append(json.loads(line))
    singleton_analysis(data)
    reverse_analysis(data)


if __name__ == "__main__":
    main()
