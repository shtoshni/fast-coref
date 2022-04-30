from typing import List, Dict, Tuple


def filter_clusters(clusters: List, threshold: int = 1) -> List:
    """Filter clusters with mentions less than the specified threshold."""

    return [
        tuple(tuple(mention) for mention in cluster)
        for cluster in clusters
        if len(cluster) >= threshold
    ]


def get_mention_to_cluster(clusters: List) -> Dict:
    """Get mention to cluster mapping."""

    clusters = [tuple(tuple(mention) for mention in cluster) for cluster in clusters]
    mention_to_cluster_dict = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster
    return mention_to_cluster_dict


def get_mention_to_cluster_idx(clusters: List) -> Dict:
    """Get mention to cluster idx mapping while filtering clustering."""

    clusters = [tuple(tuple(mention) for mention in cluster) for cluster in clusters]
    mention_to_cluster_dict = {}
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            mention_to_cluster_dict[mention] = cluster_idx
    return mention_to_cluster_dict


def is_aligned(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Return true if one of the span is a substring of the other span."""

    if span1[0] >= span2[0] and span1[1] <= span2[1]:
        return True
    if span2[0] >= span1[0] and span2[1] <= span1[1]:
        return True
    return False
