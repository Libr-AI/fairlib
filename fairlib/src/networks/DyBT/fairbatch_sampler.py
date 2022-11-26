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

class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch.

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

        # Default batch size
        self.S = {}
        
        for tmp_yg in self.yg_tuple:
            self.S[tmp_yg] = self.batch_size * (self.yg_len[tmp_yg])/self.N

        self.lb_dict = {}
        for tmp_y in self.y_item:
            for tmp_g in self.g_item:
                self.lb_dict[(tmp_y, tmp_g)] = self.yg_len[(tmp_y, tmp_g)] / self.y_len[tmp_y]

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
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are described in the paper.

        """
        epoch_loss = self.epoch_loss()

        if self.fairness_type in ['stratified_y', "EO"]:

            yhat_yg = {}
            yhat_y = {}

            for tmp_yg in self.yg_tuple:
                yhat_yg[tmp_yg] = float(torch.sum(epoch_loss[self.yg_index[tmp_yg]])) / self.yg_len[tmp_yg]
                
            for tmp_y in self.y_item:
                yhat_y[tmp_y] = float(torch.sum(epoch_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
            
            # Find the max gap group within each class
            max_index = {tmp_y:None for tmp_y in self.y_item}
            max_diff = {tmp_y:0 for tmp_y in self.y_item}
            for tmp_y in self.y_item:
                for tmp_g in self.g_item:
                    diff = abs(yhat_yg[(tmp_y, tmp_g)] - yhat_y[tmp_y])
                    if diff >= max_diff[tmp_y]:
                        max_diff[tmp_y] = diff
                        max_index[tmp_y] = tmp_g
            
            # Update lambda of groups within in each class
            
            # Determin the singe of alpha, increase if a group has a larger loss. 
            for tmp_y in self.y_item:
                if yhat_yg[(tmp_y, max_index[tmp_y])] > yhat_y[tmp_y]:
                    tmp_alpha = self.alpha
                else:
                    tmp_alpha = -1*self.alpha
            
                self.lb_dict[(tmp_y, max_index[tmp_y])] += (1 + 1/(len(self.g_item)-1)) * tmp_alpha

                for tmp_g in self.g_item:
                    self.lb_dict[(tmp_y, tmp_g)] -= (1/(len(self.g_item)-1)) * tmp_alpha

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            for tmp_y in self.y_item:
                y_lbs = [self.lb_dict[(tmp_y, _g)] for _g in self.g_item]
                # Mini value >= 0
                if min(y_lbs) < 0:
                    for tmp_g in self.g_item:
                        self.lb_dict[(tmp_y, tmp_g)] -= min(y_lbs)
                    y_lbs = [self.lb_dict[(tmp_y, _g)] for _g in self.g_item]
                # Sum to 1
                y_labs_sum = sum(y_lbs)
                if y_labs_sum != 1:
                    for tmp_g in self.g_item:
                        self.lb_dict[(tmp_y, tmp_g)] = self.lb_dict[(tmp_y, tmp_g)] / y_labs_sum

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

    
    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """
        if self.fairness_type == 'original':
            
            entire_index = torch.LongTensor([i for i in range(len(self.y_data))])
            
            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)
            
            for i in range(self.batch_num):
                yield sort_index[i]
            
        else:
        
            self.adjust_lambda() # Adjust the lambda values
            each_size = {}
            
            
            # Based on the updated lambdas, determine the size of each class in a batch
            if self.fairness_type in ['stratified_y', "EO"]:
                
                for tmp_y in self.y_item:
                    for tmp_g in self.g_item:
                        each_size[(tmp_y,tmp_g)] = round(self.lb_dict[(tmp_y,tmp_g)]*self.batch_size * (self.y_len[tmp_y])/self.N)

            # Get the indices for each batch
            sort_index_all = {}
            for _tmp_yg in self.yg_tuple:
                sort_index_all[_tmp_yg] = self.select_batch_replacement(each_size[_tmp_yg], self.yg_index[_tmp_yg], self.batch_num, self.replacement)

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