from .utils import BaseDyBT

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


class Generalized_FairBatch(BaseDyBT):
    def adjust_lambda(self):
        """Adjusts the lambda values for each group
        """
        epoch_loss = self.epoch_loss()

        epoch_loss_yz = {}
        epoch_loss_y = {}
        epoch_loss_z = {}
        
        epoch_loss_mean  = float(torch.sum(epoch_loss)) / self.N

        for tmp_yz in self.yz_tuple:
            epoch_loss_yz[tmp_yz] = float(torch.sum(epoch_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]

        for tmp_y in self.y_item:
            epoch_loss_y[tmp_y] = float(torch.sum(epoch_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]

        for tmp_z in self.z_item:
            epoch_loss_z[tmp_z] = float(torch.sum(epoch_loss[self.z_index[tmp_z]])) / self.z_len[tmp_z]

        if self.fairness_type == "joint":
            larger_loss_group = []
            smaller_loss_group = []

            # Determin the singe of alpha, increase if a group has a larger loss. 
            for tmp_yz in self.yz_tuple:
                if epoch_loss_yz[tmp_yz] > epoch_loss_mean:
                    larger_loss_group.append(tmp_yz)
                elif epoch_loss_yz[tmp_yz] < epoch_loss_mean:
                    smaller_loss_group.append(tmp_yz)
            
            for llg in larger_loss_group:
                self.lb_dict[llg] += self.alpha / len(larger_loss_group)
            for slg in smaller_loss_group:
                self.lb_dict[slg] -= self.alpha / len(smaller_loss_group)

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            all_lbs = [self.lb_dict[tmp_yz] for tmp_yz in self.yz_tuple]
            min_lb_adjustment = -1 * min([min(all_lbs), 0])
            sum_lbs = sum(all_lbs) + min_lb_adjustment * len(self.yz_tuple)
            for tmp_yz in self.yz_tuple:
                self.lb_dict[tmp_yz] = (self.lb_dict[tmp_yz]+min_lb_adjustment)/sum_lbs

        elif self.fairness_type == "y":

            larger_loss_group = []
            smaller_loss_group = []

            for tem_y in self.y_item:
                if epoch_loss_y[tem_y] > epoch_loss_mean:
                    larger_loss_group.append(tem_y)
                elif epoch_loss_y[tem_y] < epoch_loss_mean:
                    smaller_loss_group.append(tem_y)
            
            for llg in larger_loss_group:
                for tem_z in self.z_item:
                    self.lb_dict[(llg, tem_z)] += self.alpha / len(larger_loss_group) * (self.yz_len[(llg, tem_z)]/self.y_len[llg])
            for slg in smaller_loss_group:
                for tem_z in self.z_item:
                    self.lb_dict[(slg, tem_z)] -= self.alpha / len(smaller_loss_group) * (self.yz_len[(slg, tem_z)]/self.y_len[slg])

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            all_lbs = [self.lb_dict[tmp_yz] for tmp_yz in self.yz_tuple]
            min_lb_adjustment = -1 * min([min(all_lbs), 0])
            sum_lbs = sum(all_lbs) + min_lb_adjustment * len(self.yz_tuple)
            for tmp_yz in self.yz_tuple:
                self.lb_dict[tmp_yz] = (self.lb_dict[tmp_yz]+min_lb_adjustment)/sum_lbs

        elif self.fairness_type == "g":
            larger_loss_group = []
            smaller_loss_group = []

            for tem_z in self.z_item:
                if epoch_loss_z[tem_z] > epoch_loss_mean:
                    larger_loss_group.append(tem_z)
                elif epoch_loss_z[tem_z] < epoch_loss_mean:
                    smaller_loss_group.append(tem_z)
            
            for llg in larger_loss_group:
                for tem_y in self.y_item:
                    self.lb_dict[(tem_y, llg)] += self.alpha / len(larger_loss_group) * (self.yz_len[(tem_y, llg)]/self.z_len[llg])
            for slg in smaller_loss_group:
                for tem_y in self.y_item:
                    self.lb_dict[(tem_y, slg)] -= self.alpha / len(smaller_loss_group) * (self.yz_len[(tem_y, slg)]/self.z_len[slg])

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            all_lbs = [self.lb_dict[tmp_yz] for tmp_yz in self.yz_tuple]
            min_lb_adjustment = -1 * min([min(all_lbs), 0])
            sum_lbs = sum(all_lbs) + min_lb_adjustment * len(self.yz_tuple)
            for tmp_yz in self.yz_tuple:
                self.lb_dict[tmp_yz] = (self.lb_dict[tmp_yz]+min_lb_adjustment)/sum_lbs
            
        elif self.fairness_type == "stratified_y":
            for tmp_y in self.y_item:
                
                larger_loss_group = []
                smaller_loss_group = []
                
                for tmp_z in self.z_item:
                    if epoch_loss_yz[(tmp_y, tmp_z)] > epoch_loss_y[tmp_y]:
                        larger_loss_group.append(tem_z)
                    elif epoch_loss_yz[(tmp_y, tmp_z)] < epoch_loss_y[tmp_y]:
                        smaller_loss_group.append(tem_z)

                for llg in larger_loss_group:
                    self.lb_dict[(tem_y, llg)] += self.alpha / len(larger_loss_group)
                for slg in smaller_loss_group:
                    self.lb_dict[(tem_y, slg)] -= self.alpha / len(smaller_loss_group)

                # Normalize probabilities
                y_lbs = [self.lb_dict[(tmp_y, _z)] for _z in self.z_item]
                min_lb_adjustment = -1 * min([min(y_lbs), 0])
                sum_lbs = (sum(y_lbs) + min_lb_adjustment * len(self.z_item) ) / (self.y_len[tmp_y]/self.N)
                for tmp_z in self.z_item:
                    self.lb_dict[(tmp_y, tmp_z)] = (self.lb_dict[(tmp_y, tmp_z)]+min_lb_adjustment) / sum_lbs


        elif self.fairness_type == "stratified_g":
            for tmp_z in self.z_item:
                
                larger_loss_group = []
                smaller_loss_group = []
                
                for tmp_y in self.y_item:
                    if epoch_loss_yz[(tmp_y, tmp_z)] > epoch_loss_z[tmp_z]:
                        larger_loss_group.append(tem_y)
                    elif epoch_loss_yz[(tmp_y, tmp_z)] < epoch_loss_z[tmp_z]:
                        smaller_loss_group.append(tem_y)

                for llg in larger_loss_group:
                    self.lb_dict[(llg, tmp_z)] += self.alpha / len(larger_loss_group)
                for slg in smaller_loss_group:
                    self.lb_dict[(slg, tmp_z)] -= self.alpha / len(smaller_loss_group)

                # Normalize probabilities
                z_lbs = [self.lb_dict[(_y, tmp_z)] for _y in self.y_item]
                min_lb_adjustment = -1 * min([min(z_lbs), 0])
                sum_lbs = (sum(z_lbs) + min_lb_adjustment * len(self.y_item) ) / (self.z_len[tmp_z]/self.N)
                for tmp_y in self.y_item:
                    self.lb_dict[(tmp_y, tmp_z)] = (self.lb_dict[(tmp_y, tmp_z)]+min_lb_adjustment) / sum_lbs