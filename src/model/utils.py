from coref_utils.utils import get_mention_to_cluster_idx


def get_gt_actions(pred_mentions, document, mem_type_config):
    if "clusters" in document:
        # Ground truth is avaliable
        gt_clusters = document["clusters"]
        if mem_type_config.name == "unbounded":
            return get_actions_unbounded_fast(pred_mentions, gt_clusters)
        elif mem_type_config.name == "learned":
            return get_actions_learned(
                pred_mentions, document["clusters"], mem_type_config.max_ents
            )
        elif mem_type_config.name == "lru":
            return get_actions_lru(
                pred_mentions, document["clusters"], mem_type_config.max_ents
            )
    else:
        # Don't have ground truth clusters i.e. running it in the wild
        # Generate dummy actions
        return [(-1, "i")] * len(pred_mentions)


def action_sequences_to_clusters(actions, mentions):
    clusters = []
    cell_to_clusters = {}

    for mention, (cell_idx, action_type) in zip(mentions, actions):
        if action_type == "c":
            cell_to_clusters[cell_idx].append(mention)
        elif action_type == "o":
            # Overwrite
            if cell_idx in cell_to_clusters:
                # Remove the old cluster and initialize the new
                clusters.append(cell_to_clusters[cell_idx])
            cell_to_clusters[cell_idx] = [mention]
        elif action_type == "n":
            clusters.append([mention])

    for cell_idx, cluster in cell_to_clusters.items():
        clusters.append(cluster)

    return clusters


def get_actions_unbounded_fast(pred_mentions, gt_clusters):
    actions = []
    cell_counter = 0
    cluster_to_cell = {}
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)

    for idx, mention in enumerate(pred_mentions):
        if tuple(mention) not in mention_to_cluster:
            actions.append((-1, "i"))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster in cluster_to_cell:
                # Cluster is already being tracked
                actions.append((cluster_to_cell[mention_cluster], "c"))
            else:
                # Cluster is not being tracked
                # Add the mention to being tracked
                cluster_to_cell[mention_cluster] = cell_counter
                actions.append((cell_counter, "o"))
                cell_counter += 1

    return actions


def get_actions_learned(pred_mentions, gt_clusters, max_ents):
    # Useful data structures
    pred_mentions = [tuple(mention) for mention in pred_mentions]
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)

    actions = []
    cell_to_cluster = {}
    cell_to_last_used = [0 for cell in range(max_ents)]  # Initialize last usage of cell
    cluster_to_cell = {}

    # Initialize with all the mentions
    cluster_to_rem_mentions = [len(cluster) for cluster in gt_clusters]

    for mention in pred_mentions:
        used_cell_idx = None
        if mention not in mention_to_cluster:
            # Invalid mention - Mention proposal module can propose spurious spans
            actions.append((-1, "i"))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster in cluster_to_cell:
                # Cluster is already being tracked
                actions.append((cluster_to_cell[mention_cluster], "c"))
                # Update when the cell was last used
                used_cell_idx = cluster_to_cell[mention_cluster]
            else:
                # Cluster is not being tracked
                # Find the cell with the least "regret" that we can overwrite to
                # If the regret is non-positive i.e. we would be missing out on > mentions
                # of a cluster being currently tracked than the new mention cluster then we
                # don't perform overwrite.
                cur_rem_mentions = cluster_to_rem_mentions[mention_cluster]
                cell_info = []
                for cell_idx in range(max_ents):
                    if cell_idx in cell_to_cluster:
                        # The cell is actually in use
                        cell_cluster = cell_to_cluster[cell_idx]
                        cell_rem_mentions = cluster_to_rem_mentions[cell_cluster]
                    else:
                        # The cell is not in use
                        cell_rem_mentions = -1

                    cell_info.append(
                        (cell_rem_mentions, cell_to_last_used[cell_idx], cell_idx)
                    )

                # Sort the cells primarily by the number of remaining mentions
                # If the remaining mentions are tied, then compare the last used cell
                cell_info = sorted(cell_info, key=lambda x: x[0] - 1e-10 * x[1])
                min_remaining_mentions = cell_info[0][0]

                if cur_rem_mentions >= min_remaining_mentions:
                    used_cell_idx = cell_info[0][2]  # Get the cell index

                if used_cell_idx is None:
                    # Ignore the mention - No space (n)
                    actions.append((-1, "n"))
                else:
                    # Overwrite
                    actions.append((used_cell_idx, "o"))
                    # Remove the cluster to cell reference for the replacement cell
                    # Only do this if the cell was tracking anything
                    if used_cell_idx in cell_to_cluster:
                        del cluster_to_cell[cell_to_cluster[used_cell_idx]]

                    # Add the mention to being tracked
                    cluster_to_cell[mention_cluster] = used_cell_idx
                    cell_to_cluster[used_cell_idx] = mention_cluster

            # Update the cell_to_last_used index
            for cell_idx in range(max_ents):
                cell_to_last_used[cell_idx] += 1
            if used_cell_idx is not None:
                cell_to_last_used[used_cell_idx] = 0

            # Reduce the number of mentions remaining in the current cluster
            cluster_to_rem_mentions[mention_cluster] -= 1

    return actions


def get_actions_lru(pred_mentions, gt_clusters, max_ents):
    pred_mentions = [tuple(mention) for mention in pred_mentions]

    # Useful data structures
    mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)

    actions = []
    cell_to_cluster = {}
    cell_to_last_used = [0 for cell in range(max_ents)]  # Initialize last usage of cell
    cluster_to_cell = {}

    # Initialize with all the mentions
    # cluster_to_rem_mentions = [len(cluster) for cluster in clusters]
    cluster_to_rem_mentions = [len(cluster) for cluster in gt_clusters]
    lru_list = list(range(max_ents))

    for mention in pred_mentions:
        used_cell_idx = None
        if mention not in mention_to_cluster:
            # Not a mention
            actions.append((-1, "i"))
        else:
            mention_cluster = mention_to_cluster[tuple(mention)]
            if mention_cluster in cluster_to_cell:
                # Cluster is already being tracked
                actions.append((cluster_to_cell[mention_cluster], "c"))
                # Update when the cell was last used
                used_cell_idx = cluster_to_cell[mention_cluster]
            else:
                # Cluster is not being tracked
                # Find the cell with the least regret that we can overwrite to
                # If the regret is non-positive i.e. we would be missing out on >= mentions
                # of a cluster being currently tracked than the new mention cluster then we
                # don't perform overwrite.
                cur_rem_mentions = cluster_to_rem_mentions[mention_cluster]
                cell_info = []
                for cell_idx in range(max_ents):
                    if cell_idx in cell_to_cluster:
                        # The cell is actually in use
                        cell_cluster = cell_to_cluster[cell_idx]
                        cell_rem_mentions = cluster_to_rem_mentions[cell_cluster]
                    else:
                        # The cell is not in use
                        cell_rem_mentions = -1
                    cell_info.append(
                        (
                            cell_rem_mentions,
                            cell_to_last_used[cell_idx],
                            cell_idx,
                            lru_list.index(cell_idx),
                        )
                    )

                # Sort cells by least recently used cells
                cell_info = sorted(cell_info, key=lambda x: x[3])

                # Remaining mentions in least recently used cell
                lru_remaining_mentions = cell_info[0][0]

                if cur_rem_mentions >= lru_remaining_mentions:
                    used_cell_idx = cell_info[0][2]  # Get the cell index

                if used_cell_idx is None:
                    # Ignore the mention
                    actions.append((-1, "n"))
                else:
                    # Overwrite
                    actions.append((used_cell_idx, "o"))
                    # Remove the cluster to cell reference for the replacement cell
                    # Only do this if the cell was tracking anything
                    if used_cell_idx in cell_to_cluster:
                        del cluster_to_cell[cell_to_cluster[used_cell_idx]]

                    # Add the mention to being tracked
                    cluster_to_cell[mention_cluster] = used_cell_idx
                    cell_to_cluster[used_cell_idx] = mention_cluster

            # Update the cell_to_last_used index
            for cell_idx in range(max_ents):
                cell_to_last_used[cell_idx] += 1
            if used_cell_idx is not None:
                cell_to_last_used[used_cell_idx] = 0
                # Remove the used_cell_idx and put it at the end of the LRU list
                lru_list.remove(used_cell_idx)
                lru_list.append(used_cell_idx)

            # Reduce the number of mentions remaining in the current cluster
            cluster_to_rem_mentions[mention_cluster] -= 1

    return actions
