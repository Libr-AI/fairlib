import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

import logging

class BaseDyBT(Sampler):
    """Base Sample Class (Sampler in DataLoader).
    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, g_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type
                        stratified_y : equalized odds (eqodds) in binary and equal opportunity (eqopp) in multi-class classification
                        stratified_g : demographic parity
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, g_item: Lists that contains the unique values of the y_data and g_data, respectively.
        yg_tuple: Lists for pairs of y_item and g_item.
        y_, g_, yg_mask: Dictionaries utilizing as array masks.
        y_, g_, yg_index: Dictionaries containing the index of each class.
        y_, g_, yg_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb_dict: A dictionary of real numbers indicating the lambda values in FairBatch.
        
    """
    def __init__(self, model, args, replacement=False):
        """Initializes FairBatch."""
        
        self.model = model
        np.random.seed(args.base_seed)
        random.seed(args.base_seed)
        
        self.args = args

        train_dataset = args.train_generator.dataset
        eval_dataloader_params = {
            'batch_size': args.test_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers}
        self.data_iterator = torch.utils.data.DataLoader(train_dataset, **eval_dataloader_params)

        self.x_data = torch.from_numpy(train_dataset.X)
        self.y_data = torch.from_numpy(train_dataset.y)
        self.g_data = torch.from_numpy(train_dataset.protected_label)
        
        self.alpha = args.DyBTalpha
        self.fairness_type = args.DyBTObj
        self.replacement = replacement
        
        self.N = len(self.g_data)
        
        self.batch_size = args.batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the target and group labels
        self.g_item = list(set(self.g_data.tolist()))
        self.y_item = list(set(self.y_data.tolist()))
        
        self.yg_tuple = list(itertools.product(self.y_item, self.g_item))
        
        # Makes masks
        self.g_mask = {}
        self.y_mask = {}
        self.yg_mask = {}
        
        for tmp_g in self.g_item:
            self.g_mask[tmp_g] = (self.g_data == tmp_g)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yg in self.yg_tuple:
            self.yg_mask[tmp_yg] = (self.y_data == tmp_yg[0]) & (self.g_data == tmp_yg[1])
        

        # Finds the index
        self.g_index = {}
        self.y_index = {}
        self.yg_index = {}
        
        for tmp_g in self.g_item:
            self.g_index[tmp_g] = (self.g_mask[tmp_g] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yg in self.yg_tuple:
            self.yg_index[tmp_yg] = (self.yg_mask[tmp_yg] == 1).nonzero().squeeze()
            
        # Length information
        self.g_len = {}
        self.y_len = {}
        self.yg_len = {}
        
        for tmp_g in self.g_item:
            self.g_len[tmp_g] = len(self.g_index[tmp_g])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yg in self.yg_tuple:
            self.yg_len[tmp_yg] = len(self.yg_index[tmp_yg])

        
        self.lb_dict = {}
        assert args.DyBTinit in ["original", "balanced"], NotImplemented
        if args.DyBTinit == "original":
            # Default batch size
            for tmp_yg in self.yg_tuple:
                self.lb_dict[tmp_yg] = (self.yg_len[tmp_yg])/self.N
        else:
            if args.DyBTObj == "joint":
                for tmp_yg in self.yg_tuple:
                    self.lb_dict[tmp_yg] = (1.0/len(self.yg_tuple))

            elif args.DyBTObj == "y":
                for tmp_yg in self.yg_tuple:
                    self.lb_dict[tmp_yg] = (self.yg_len[tmp_yg]/self.y_len[tmp_yg[0]]) * (1.0/len(self.y_item))

            elif args.DyBTObj == "g":
                for tmp_yg in self.yg_tuple:
                    self.lb_dict[tmp_yg] = (self.yg_len[tmp_yg]/self.g_len[tmp_yg[1]]) * (1.0/len(self.g_item))

            elif args.DyBTObj in ["stratified_y", "EO"]:
                for tmp_yg in self.yg_tuple:
                    self.lb_dict[tmp_yg] = self.y_len[tmp_yg[0]]/self.N * (1.0/len(self.g_item))

            elif args.DyBTObj == "stratified_g":
                for tmp_yg in self.yg_tuple:
                    self.lb_dict[tmp_yg] = self.g_len[tmp_yg[1]]/self.N * (1.0/len(self.y_item))
            
            else:
                raise NotImplementedError

    def epoch_loss(self):
        device = self.args.device
        
        self.model.eval()

        if self.args.regression:
            criterion = torch.nn.MSELoss(reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

        batch_losses = []

        for batch in self.data_iterator:
            
            text = batch[0].squeeze()
            tags = batch[1].squeeze()
            p_tags = batch[2].squeeze()

            text = text.to(device)
            tags = tags.to(device).long()
            p_tags = p_tags.to(device).float()

            if self.args.BT is not None and self.args.BT == "Reweighting":
                instance_weights = batch[3].float()
                instance_weights = instance_weights.to(device)

            if self.args.regression:
                regression_tags = batch[5].squeeze()
                regression_tags = regression_tags.to(device)

            # main model predictions
            if self.args.gated:
                predictions = self.model(text, p_tags)
            else:
                predictions = self.model(text)

            predictions = predictions if not self.args.regression else predictions.squeeze()

            # add the weighted loss
            if self.args.BT is not None and self.args.BT == "Reweighting":
                loss = criterion(predictions, tags)
                loss = loss * instance_weights
            else:
                loss = criterion(predictions, tags)

            batch_losses.append(loss.detach().cpu())
        
        return torch.cat(batch_losses, dim=0)

    def adjust_lambda(self):
        """Adjusts the lambda values for each group
        """
        raise NotImplementedError


    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    
                    start_idx = len(full_index)-start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
            
        return select_index

    def update_size_of_all_group(self):
        each_size = {}

        for tmp_yg in self.yg_tuple:
            each_size[tmp_yg] = round(self.lb_dict[tmp_yg]*self.batch_size)

        return each_size

    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """
        self.adjust_lambda() # Adjust the lambda values

        each_size = self.update_size_of_all_group()

        # Get the indices for each batch
        sort_index_all = {}
        for _tmp_yg in self.yg_tuple:
            sort_index_all[_tmp_yg] = self.select_batch_replacement(each_size[(_tmp_yg)], self.yg_index[(_tmp_yg)], self.batch_num, self.replacement)

        for t in range(self.batch_num):
            key_in_fairbatch = []
            for tmp_yg in self.yg_tuple:
                key_in_fairbatch.append(sort_index_all[tmp_yg][t].copy())

            key_in_fairbatch = np.hstack(key_in_fairbatch)

            random.shuffle(key_in_fairbatch)
            yield key_in_fairbatch

    def __len__(self):
        """Returns the number of batch."""
        
        return self.batch_num