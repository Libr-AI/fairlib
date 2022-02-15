import logging
from typing import Dict

import numpy as np

import torch
import torch.utils.data as data

from collections import Counter
from random import shuffle

def full_label_data(df, tasks):
    selected_rows = np.array([True]*len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.X = []
        self.y = []
        self.protected_label = []
        self.instance_weights = []

        self.load_data()
        
        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y)
        self.protected_label = np.array(self.protected_label)

        self.balanced_training()

        print("Loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index]
    
    def load_data(self):
        pass

    def balanced_training(self):
        if self.args.BT is None:
            # Without balanced training
            self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.BT in ["Reweighting", "Resampling"], "not implemented"

            assert self.args.BTObj in ["joint", "y", "g", "stratified_y", "stratified_g"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y)
                stratified_g:   balancing the y for each g, p(y|g)
            """

            if self.args.BT == "Reweighting":
                # init a dict for storing the index of each group.

                n_total = len(self.y)
                if self.args.BTObj in ["joint", "stratified_y", "stratified_g"]:
                    weighting_counter = Counter([(i,j) for i,j in zip(self.y, self.protected_label)])
                elif self.args.BTObj == "y":
                    weighting_counter = Counter(self.y)
                elif self.args.BTObj == "g":
                    weighting_counter = Counter(self.protected_label)
                else:
                    pass

                if self.args.BTObj in ["joint", "y", "g"]:
                    n_perfect_balanced = n_total / len(weighting_counter.keys())
                    for k in weighting_counter.keys():
                        weighting_counter[k] = n_perfect_balanced / weighting_counter[k]
                elif self.args.BTObj == "stratified_y":
                    for k in weighting_counter.keys():
                        _y, _g = k
                        groups_with_same_y = [_k for _k in weighting_counter.keys() if _k[0] == _y]
                        num_y = sum([weighting_counter[_k] for _k in groups_with_same_y])
                        weighting_counter[k] = num_y / len(groups_with_same_y) / weighting_counter[k]
                elif self.args.BTObj == "stratified_g":
                    for k in weighting_counter.keys():
                        _y, _g = k
                        groups_with_same_g = [_k for _k in weighting_counter.keys() if _k[1] == _g]
                        num_g = sum([weighting_counter[_k] for _k in groups_with_same_g])
                        weighting_counter[k] = num_g / len(groups_with_same_g) / weighting_counter[k]
                else:
                    pass

                # add weights
                self.instance_weights = []
                for _y, _g in zip(self.y, self.protected_label):
                    if self.args.BTObj in ["joint", "stratified_y", "stratified_g"]:
                        self.instance_weights.append(weighting_counter[(_y, _g)])
                    elif self.args.BTObj == "y":
                        self.instance_weights.append(weighting_counter[_y])
                    elif self.args.BTObj == "g":
                        self.instance_weights.append(weighting_counter[_g])
                    else:
                        pass

                self.instance_weights = np.array(self.instance_weights)

            elif self.args.BT == "Resampling":
                
                # init a dict for storing the index of each group.

                group_idx = {}
                if self.args.BTObj in ["joint", "stratified_y", "stratified_g"]:
                    group_labels = [(i,j) for i,j in zip(self.y, self.protected_label)]
                elif self.args.BTObj == "y":
                    group_labels = self.y
                elif self.args.BTObj == "g":
                    group_labels = self.protected_label
                else:
                    pass

                for idx, group_label in enumerate(group_labels):
                    group_idx[group_label] = group_idx.get(group_label, []) + [idx]

                selected_index = []

                if self.args.BTObj in ["joint", "y", "g"]:
                    selected = min([len(i) for i in group_idx.values()])

                    for index in group_idx.values():
                        _index = index
                        shuffle(_index)
                        selected_index = selected_index + _index[:selected]

                elif self.args.BTObj == "stratified_y":
                    # empirical distribution of y
                    weighting_counter = Counter(self.y)

                    # a list of (weights, actual length)
                    condidate_selected = min([len(group_idx[(_y, _g)])/weighting_counter[_y] for (_y, _g) in group_idx.keys()])

                    distinct_y_label = set(self.y)
                    distinct_g_label = set(self.protected_label)
                    
                    # iterate each main task class
                    for y in distinct_y_label:
                        selected = int(condidate_selected * weighting_counter[y])
                        for g in distinct_g_label:
                            _index = group_idx[(y,g)]
                            shuffle(_index)
                            selected_index = selected_index + _index[:selected]

                elif self.args.BTObj == "stratified_g":
                    # empirical distribution of g
                    weighting_counter = Counter(self.protected_label)

                    # a list of (weights, actual length)
                    # Noticing that if stratified_g, the order within the key has been changed.
                    condidate_selected = min([len(group_idx[(_y, _g)])/weighting_counter[_g] for (_y, _g) in group_idx.keys()])

                    distinct_y_label = set(self.y)
                    distinct_g_label = set(self.protected_label)
                    
                    # iterate each main task class
                    # for y in distinct_y_label:
                    for g in distinct_g_label:
                        selected = int(condidate_selected * weighting_counter[g])
                        # for g in distinct_g_label:
                        for y in distinct_y_label:
                            _index = group_idx[(y,g)]
                            shuffle(_index)
                            selected_index = selected_index + _index[:selected]

                X = [self.X[index] for index in selected_index]
                self.X = np.array(X)
                y = [self.y[index] for index in selected_index]
                self.y = np.array(y)
                gender_label = [self.protected_label[index] for index in selected_index]
                self.protected_label = np.array(gender_label)
                self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])

            else:
                pass
        return None