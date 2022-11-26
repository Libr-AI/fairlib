import numpy as np
import torch
import pandas as pd
from .BT import get_weights, get_sampled_indices
from .generalized_BT import get_data_distribution, manipulate_data_distribution

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

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.X = []
        self.y = []
        self.protected_label = []
        self.instance_weights = []
        self.adv_instance_weights = []
        self.regression_label = []
        self.addition_values = {}

        self.load_data()

        self.regression_init()
        
        self.X = np.array(self.X)
        if len(self.X.shape) == 3:
            self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y).astype(int)
        self.protected_label = np.array(self.protected_label).astype(int)

        self.manipulate_data_distribution()

        self.balanced_training()

        self.adv_balanced_training()

        if self.split == "train":
            self.adv_decoupling()

        print("Loaded data shapes: {}, {}, {}".format(self.X.shape, self.y.shape, self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        _X = self.X[index]
        _y = self.y[index]
        _protected_label = self.protected_label[index]
        _instance_weights = self.instance_weights[index]
        _adv_instance_weights = self.adv_instance_weights[index]
        _regression_label = self.regression_label[index]

        data_dict = {
            0:_X,
            1:_y,
            2:_protected_label,
            3:_instance_weights,
            4:_adv_instance_weights,
            5:_regression_label,
        }
        for _k in self.addition_values.keys():
            if _k not in data_dict.keys():
                data_dict[_k] = self.addition_values[_k][index]
        return data_dict
    
    def load_data(self):
        pass

    def manipulate_data_distribution(self):
        if self.args.GBT and self.split == "train":
            # Get data distribution
            distribution_dict = get_data_distribution(y_data=self.y, g_data=self.protected_label)

            selected_index = manipulate_data_distribution(
                default_distribution_dict = distribution_dict, 
                N = self.args.GBT_N, 
                GBTObj = self.args.GBTObj, 
                alpha = self.args.GBT_alpha)

            self.X = self.X[selected_index]
            self.y = self.y[selected_index]
            self.protected_label = self.protected_label[selected_index]

            for _k in self.addition_values.keys():
                self.addition_values[_k] = [self.addition_values[_k][index] for index in selected_index]

    def balanced_training(self):
        if (self.args.BT is None) or (self.split != "train"):
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
                _protected_label = [self.protected_label[index] for index in selected_index]
                self.protected_label = np.array(_protected_label)
                self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])

                for _k in self.addition_values.keys():
                    self.addition_values[_k] = [self.addition_values[_k][index] for index in selected_index]

            else:
                raise NotImplementedError
        return None

    def adv_balanced_training(self):
        if (self.args.adv_BT is None) or (self.split != "train"):
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

    def adv_decoupling(self):
        """Simulating unlabelled protected labels through assigning -1 to instances.

        Returns:
            None
        """
        if self.args.adv_decoupling and self.args.adv_decoupling_labelled_proportion < 1:
            self.adv_instance_weights[
                np.random.rand(len(self.protected_label)) > self.args.adv_decoupling_labelled_proportion
                ] = -1
        else:
            pass
        return None
    
    def regression_init(self):
        if not self.args.regression:
            self.regression_label = np.array([0 for _ in range(len(self.protected_label))])
        else:
            # Discretize variable into equal-sized buckets
            if self.split == "train":
                bin_labels, bins = pd.qcut(self.y, q=self.args.n_bins, labels=False, duplicates = "drop", retbins = True)
                self.args.regression_bins = bins
            else:
                bin_labels = pd.cut(self.y, bins=self.args.regression_bins, labels=False, duplicates = "drop", include_lowest = True)
            bin_labels = np.nan_to_num(bin_labels, nan=0)
            
            # Reassign labels
            self.regression_label, self.y = np.array(self.y), bin_labels