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
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type
                        stratified_y : equalized odds (eqodds) in binary and equal opportunity (eqopp) in multi-class classification
                        stratified_g : demographic parity
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
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
        self.z_data = torch.from_numpy(train_dataset.protected_label)
        
        self.alpha = args.DyBTalpha
        self.fairness_type = args.DyBTObj
        self.replacement = replacement
        
        self.N = len(self.z_data)
        
        self.batch_size = args.batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)
        
        # Takes the unique values of the target and group labels
        self.z_item = list(set(self.z_data.tolist()))
        self.y_item = list(set(self.y_data.tolist()))
        
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}
        
        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)
            
        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)
            
        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])
        

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}
        
        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()
            
        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()
        
        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()
            
        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}
        
        for tmp_z in self.z_item:
            self.z_len[tmp_z] = len(self.z_index[tmp_z])
            
        for tmp_y in self.y_item:
            self.y_len[tmp_y] = len(self.y_index[tmp_y])
            
        for tmp_yz in self.yz_tuple:
            self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        
        self.lb_dict = {}
        assert args.DyBTinit in ["original", "balanced"], NotImplemented
        if args.DyBTinit == "original":
            # Default batch size
            for tmp_yz in self.yz_tuple:
                self.lb_dict[tmp_yz] = (self.yz_len[tmp_yz])/self.N
        else:
            if args.DyBTObj == "joint":
                for tmp_yz in self.yz_tuple:
                    self.lb_dict[tmp_yz] = (1.0/len(self.yz_tuple))

            elif args.DyBTObj == "y":
                for tmp_yz in self.yz_tuple:
                    self.lb_dict[tmp_yz] = (self.yz_len[tmp_yz]/self.y_len[tmp_yz[0]]) * (1.0/len(self.y_item))

            elif args.DyBTObj == "g":
                for tmp_yz in self.yz_tuple:
                    self.lb_dict[tmp_yz] = (self.yz_len[tmp_yz]/self.z_len[tmp_yz[1]]) * (1.0/len(self.z_item))

            elif args.DyBTObj == "stratified_y":
                for tmp_yz in self.yz_tuple:
                    self.lb_dict[tmp_yz] = self.y_len[tmp_yz[0]]/self.N * (1.0/len(self.z_item))

            elif args.DyBTObj == "stratified_g":
                for tmp_yz in self.yz_tuple:
                    self.lb_dict[tmp_yz] = self.z_len[tmp_yz[1]]/self.N * (1.0/len(self.y_item))

    def epoch_loss(self):
        device = self.args.device
        
        self.model.eval()

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

            # main model predictions
            if self.args.gated:
                predictions = self.model(text, p_tags)
            else:
                predictions = self.model(text)

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

        for tmp_yz in self.yz_tuple:
            each_size[tmp_yz] = round(self.lb_dict[tmp_yz]*self.batch_size)

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
        for _tmp_yz in self.yz_tuple:
            sort_index_all[_tmp_yz] = self.select_batch_replacement(each_size[(_tmp_yz)], self.yz_index[(_tmp_yz)], self.batch_num, self.replacement)

        for t in range(self.batch_num):
            key_in_fairbatch = []
            for tmp_yz in self.yz_tuple:
                key_in_fairbatch.append(sort_index_all[tmp_yz][t].copy())

            key_in_fairbatch = np.hstack(key_in_fairbatch)

            random.shuffle(key_in_fairbatch)
            yield key_in_fairbatch

    def __len__(self):
        """Returns the number of batch."""
        
        return self.batch_num