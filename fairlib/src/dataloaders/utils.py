import logging
from typing import Dict

import numpy as np

import torch
import torch.utils.data as data

from collections import Counter
from random import shuffle
from random import choices

def full_label_data(df, tasks):
    """filter the instances with all required labels

    Args:
        df (pd.DataFrame): a DataFrame containing data instances
        tasks (list): a list of names of target columns

    Returns:
        np.array: an array of boolean values indicating whether or not each row meets the requirement.
    """
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows

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
        list: a list of indices of selected instacnes.
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
        condidate_selected = min([len(group_idx[(_y, _g)])/weighting_counter[_y] for (_y, _g) in group_idx.keys()])

        distinct_y_label = set(y)
        distinct_g_label = set(protected_label)

        # iterate each main task class
        for y in distinct_y_label:
            if method == "Downsampling":
                selected = int(condidate_selected * weighting_counter[y])
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
        condidate_selected = min([len(group_idx[(_y, _g)])/weighting_counter[_g] for (_y, _g) in group_idx.keys()])

        distinct_y_label = set(y)
        distinct_g_label = set(protected_label)

        # iterate each main task class
        # for y in distinct_y_label:
        for g in distinct_g_label:
            if method == "Downsampling":
                selected = int(condidate_selected * weighting_counter[g])
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

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.X = []
        self.y = []
        self.protected_label = []
        self.instance_weights = []
        self.adv_instance_weights = []

        self.load_data()
        
        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y)
        self.protected_label = np.array(self.protected_label)

        self.balanced_training()

        self.adv_balanced_training()

        print("Loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index], self.adv_instance_weights[index]
    
    def load_data(self):
        pass

    def balanced_training(self):
        if self.args.BT is None:
            # Without balanced training
            self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.BT in ["Reweighting", "Resampling", "Downsampling"], "not implemented"

            assert self.args.BTObj in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y), while keeping the y distribution
                stratified_g:   balancing the y for each g, p(y|g)
                EO:             balancing the g for each y, p(g|y)
            """

            if self.args.BT == "Reweighting":
                self.instance_weights = get_weights(self.args.BTObj, self.y, self.protected_label)

            elif self.args.BT in ["Resampling", "Downsampling"]:

                selected_index = get_sampled_indices(self.args.BTObj, self.y, self.protected_label, method = self.args.BT)

                X = [self.X[index] for index in selected_index]
                self.X = np.array(X)
                y = [self.y[index] for index in selected_index]
                self.y = np.array(y)
                gender_label = [self.protected_label[index] for index in selected_index]
                self.protected_label = np.array(gender_label)
                self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])

            else:
                raise NotImplementedError
        return None

    def adv_balanced_training(self):
        if self.args.adv_BT is None:
            # Without balanced training
            self.adv_instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.adv_BT in ["Reweighting"], "not implemented"

            assert self.args.adv_BTObj in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y)
                stratified_g:   balancing the y for each g, p(y|g)
            """

            if self.args.adv_BT == "Reweighting":
                self.adv_instance_weights = get_weights(self.args.adv_BTObj, self.y, self.protected_label)
            else:
                raise NotImplementedError
        return None