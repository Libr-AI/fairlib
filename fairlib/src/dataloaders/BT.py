import numpy as np
from collections import Counter
from random import shuffle
from random import choices

def get_weights(BTObj, y, protected_label):
    """Given the balanced training objective, target labels, and protected labels, pre-calculate weights for each instance.

    Args:
        BTObj (str): y| g | joint | stratified_y | stratified_g | EO.
        y (list): a list of target labels.
        protected_label (list): a list of protected labels.

    Returns:
        list: instance weights w.r.t. the balanced training objective.
    """
    # init a dict for storing the index of each group.

    n_total = len(y)
    if BTObj in ["joint", "stratified_y", "stratified_g", "EO"]:
        weighting_counter = Counter([(i,j) for i,j in zip(y, protected_label)])
    elif BTObj == "y":
        weighting_counter = Counter(y)
    elif BTObj == "g":
        weighting_counter = Counter(protected_label)
    else:
        pass

    if BTObj in ["joint", "y", "g"]:
        n_perfect_balanced = n_total / len(weighting_counter.keys())
        for k in weighting_counter.keys():
            weighting_counter[k] = n_perfect_balanced / weighting_counter[k]
    elif BTObj == "EO":
        for k in weighting_counter.keys():
            _y, _g = k                    
            groups_with_same_y = [_k for _k in weighting_counter.keys() if _k[0] == _y]
            num_y = sum([weighting_counter[_k] for _k in groups_with_same_y])
            weighting_counter[k] = num_y / weighting_counter[k]
    elif BTObj == "stratified_y":
        for k in weighting_counter.keys():
            _y, _g = k
            groups_with_same_y = [_k for _k in weighting_counter.keys() if _k[0] == _y]
            num_y = sum([weighting_counter[_k] for _k in groups_with_same_y])
            weighting_counter[k] = num_y / len(groups_with_same_y) / weighting_counter[k]
    elif BTObj == "stratified_g":
        for k in weighting_counter.keys():
            _y, _g = k
            groups_with_same_g = [_k for _k in weighting_counter.keys() if _k[1] == _g]
            num_g = sum([weighting_counter[_k] for _k in groups_with_same_g])
            weighting_counter[k] = num_g / len(groups_with_same_g) / weighting_counter[k]
    else:
        pass

    # add weights
    instance_weights = []
    for _y, _g in zip(y, protected_label):
        if BTObj in ["joint", "stratified_y", "stratified_g", "EO"]:
            instance_weights.append(weighting_counter[(_y, _g)])
        elif BTObj == "y":
            instance_weights.append(weighting_counter[_y])
        elif BTObj == "g":
            instance_weights.append(weighting_counter[_g])
        else:
            pass
    instance_weights = np.array(instance_weights)

    return instance_weights

def get_sampled_indices(BTObj, y, protected_label, method = "Downsampling"):
    """Given the balanced training objective, target labels, and protected labels, sampling instances for each group.

    Args:
        BTObj (str): y| g | joint | stratified_y | stratified_g | EO.
        y (list): a list of target labels.
        protected_label (list): a list of protected labels.
        method (str, optional): Downsampling | Resampling. Defaults to "Downsampling".

    Returns:
        list: a list of indices of selected instances.
    """

    # init a dict for storing the index of each group.
    group_idx = {}
    if BTObj in ["joint", "stratified_y", "stratified_g", "EO"]:
        group_labels = [(i,j) for i,j in zip(y, protected_label)]
    elif BTObj == "y":
        group_labels = y
    elif BTObj == "g":
        group_labels = protected_label
    else:
        pass

    for idx, group_label in enumerate(group_labels):
        group_idx[group_label] = group_idx.get(group_label, []) + [idx]

    selected_index = []

    if BTObj in ["joint", "y", "g"]:
        if method == "Downsampling":
            selected = min([len(i) for i in group_idx.values()])
        elif method == "Resampling":
            selected = (len(y) / len(group_idx.keys()))
        for index in group_idx.values():
            _index = index
            if method == "Downsampling":
                shuffle(_index)
                selected_index = selected_index + _index[:selected]
            elif method == "Resampling":
                selected_index = selected_index + choices(_index, k=int(selected))

    elif BTObj == "EO":
        # a list of (weights, actual length)
        distinct_y_label = set(y)
        distinct_g_label = set(protected_label)

        # iterate each main task class
        for y in distinct_y_label:
            if method == "Downsampling":
                selected = min([len(group_idx.get((y, _g))) for _g in distinct_g_label])
            elif method == "Resampling":
                selected = sum([len(group_idx.get((y, _g))) for _g in distinct_g_label]) / len(distinct_g_label)

            for g in distinct_g_label:
                _index = group_idx.get((y,g), [])
                if method == "Downsampling":
                    shuffle(_index)
                    selected_index = selected_index + _index[:selected]
                elif method == "Resampling":
                    selected_index = selected_index + choices(_index, k=int(selected))

    elif BTObj == "stratified_y":
        # empirical distribution of y
        weighting_counter = Counter(y)

        # a list of (weights, actual length)
        candidate_selected = min([len(group_idx[(_y, _g)])/weighting_counter[_y] for (_y, _g) in group_idx.keys()])

        distinct_y_label = set(y)
        distinct_g_label = set(protected_label)

        # iterate each main task class
        for y in distinct_y_label:
            if method == "Downsampling":
                selected = int(candidate_selected * weighting_counter[y])
            elif method == "Resampling":
                selected = int(weighting_counter[y] / len(distinct_g_label))
            for g in distinct_g_label:
                _index = group_idx.get((y,g), [])
                if method == "Downsampling":
                    shuffle(_index)
                    selected_index = selected_index + _index[:selected]
                elif method == "Resampling":
                    selected_index = selected_index + choices(_index, k=int(selected))

    elif BTObj == "stratified_g":
        # empirical distribution of g
        weighting_counter = Counter(protected_label)
        # a list of (weights, actual length)
        # Noticing that if stratified_g, the order within the key has been changed.
        candidate_selected = min([len(group_idx[(_y, _g)])/weighting_counter[_g] for (_y, _g) in group_idx.keys()])

        distinct_y_label = set(y)
        distinct_g_label = set(protected_label)

        # iterate each main task class
        # for y in distinct_y_label:
        for g in distinct_g_label:
            if method == "Downsampling":
                selected = int(candidate_selected * weighting_counter[g])
            elif method == "Resampling":
                selected = int(weighting_counter[g] / len(distinct_y_label))
            # for g in distinct_g_label:
            for y in distinct_y_label:
                _index = group_idx.get((y,g), [])
                if method == "Downsampling":
                    shuffle(_index)
                    selected_index = selected_index + _index[:selected]
                elif method == "Resampling":
                    selected_index = selected_index + choices(_index, k=int(selected))
    return selected_index