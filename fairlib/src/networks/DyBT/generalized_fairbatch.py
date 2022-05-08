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

        epoch_loss_yg = {}
        epoch_loss_y = {}
        epoch_loss_g = {}
        
        epoch_loss_mean  = float(torch.sum(epoch_loss)) / self.N

        for tmp_yg in self.yg_tuple:
            epoch_loss_yg[tmp_yg] = float(torch.sum(epoch_loss[self.yg_index[tmp_yg]])) / self.yg_len[tmp_yg]

        for tmp_y in self.y_item:
            epoch_loss_y[tmp_y] = float(torch.sum(epoch_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]

        for tmp_g in self.g_item:
            epoch_loss_g[tmp_g] = float(torch.sum(epoch_loss[self.g_index[tmp_g]])) / self.g_len[tmp_g]

        if self.fairness_type == "joint":
            larger_loss_group = []
            smaller_loss_group = []

            # Determin the singe of alpha, increase if a group has a larger loss. 
            for tmp_yg in self.yg_tuple:
                if epoch_loss_yg[tmp_yg] > epoch_loss_mean:
                    larger_loss_group.append(tmp_yg)
                elif epoch_loss_yg[tmp_yg] < epoch_loss_mean:
                    smaller_loss_group.append(tmp_yg)
            
            for llg in larger_loss_group:
                self.lb_dict[llg] += self.alpha / len(larger_loss_group)
            for slg in smaller_loss_group:
                self.lb_dict[slg] -= self.alpha / len(smaller_loss_group)

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            all_lbs = [self.lb_dict[tmp_yg] for tmp_yg in self.yg_tuple]
            min_lb_adjustment = -1 * min([min(all_lbs), 0])
            sum_lbs = sum(all_lbs) + min_lb_adjustment * len(self.yg_tuple)
            for tmp_yg in self.yg_tuple:
                self.lb_dict[tmp_yg] = (self.lb_dict[tmp_yg]+min_lb_adjustment)/sum_lbs

        elif self.fairness_type == "y":

            larger_loss_group = []
            smaller_loss_group = []

            for tem_y in self.y_item:
                if epoch_loss_y[tem_y] > epoch_loss_mean:
                    larger_loss_group.append(tem_y)
                elif epoch_loss_y[tem_y] < epoch_loss_mean:
                    smaller_loss_group.append(tem_y)
            
            for llg in larger_loss_group:
                for tem_g in self.g_item:
                    self.lb_dict[(llg, tem_g)] += self.alpha / len(larger_loss_group) * (self.yg_len[(llg, tem_g)]/self.y_len[llg])
            for slg in smaller_loss_group:
                for tem_g in self.g_item:
                    self.lb_dict[(slg, tem_g)] -= self.alpha / len(smaller_loss_group) * (self.yg_len[(slg, tem_g)]/self.y_len[slg])

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            all_lbs = [self.lb_dict[tmp_yg] for tmp_yg in self.yg_tuple]
            min_lb_adjustment = -1 * min([min(all_lbs), 0])
            sum_lbs = sum(all_lbs) + min_lb_adjustment * len(self.yg_tuple)
            for tmp_yg in self.yg_tuple:
                self.lb_dict[tmp_yg] = (self.lb_dict[tmp_yg]+min_lb_adjustment)/sum_lbs

        elif self.fairness_type == "g":
            larger_loss_group = []
            smaller_loss_group = []

            for tem_g in self.g_item:
                if epoch_loss_g[tem_g] > epoch_loss_mean:
                    larger_loss_group.append(tem_g)
                elif epoch_loss_g[tem_g] < epoch_loss_mean:
                    smaller_loss_group.append(tem_g)
            
            for llg in larger_loss_group:
                for tem_y in self.y_item:
                    self.lb_dict[(tem_y, llg)] += self.alpha / len(larger_loss_group) * (self.yg_len[(tem_y, llg)]/self.g_len[llg])
            for slg in smaller_loss_group:
                for tem_y in self.y_item:
                    self.lb_dict[(tem_y, slg)] -= self.alpha / len(smaller_loss_group) * (self.yg_len[(tem_y, slg)]/self.g_len[slg])

            # Normalize to probability, i.e., between 0 and 1, and sum to 1
            all_lbs = [self.lb_dict[tmp_yg] for tmp_yg in self.yg_tuple]
            min_lb_adjustment = -1 * min([min(all_lbs), 0])
            sum_lbs = sum(all_lbs) + min_lb_adjustment * len(self.yg_tuple)
            for tmp_yg in self.yg_tuple:
                self.lb_dict[tmp_yg] = (self.lb_dict[tmp_yg]+min_lb_adjustment)/sum_lbs
            
        elif self.fairness_type in ["stratified_y", "EO"]:
            for tmp_y in self.y_item:
                
                larger_loss_group = []
                smaller_loss_group = []
                
                for tmp_g in self.g_item:
                    if epoch_loss_yg[(tmp_y, tmp_g)] > epoch_loss_y[tmp_y]:
                        larger_loss_group.append(tmp_g)
                    elif epoch_loss_yg[(tmp_y, tmp_g)] < epoch_loss_y[tmp_y]:
                        smaller_loss_group.append(tmp_g)

                for llg in larger_loss_group:
                    self.lb_dict[(tmp_y, llg)] += self.alpha / len(larger_loss_group)
                for slg in smaller_loss_group:
                    self.lb_dict[(tmp_y, slg)] -= self.alpha / len(smaller_loss_group)

                # Normalize probabilities
                y_lbs = [self.lb_dict[(tmp_y, _g)] for _g in self.g_item]
                min_lb_adjustment = -1 * min([min(y_lbs), 0])
                sum_lbs = (sum(y_lbs) + min_lb_adjustment * len(self.g_item) ) / (self.y_len[tmp_y]/self.N)
                for tmp_g in self.g_item:
                    self.lb_dict[(tmp_y, tmp_g)] = (self.lb_dict[(tmp_y, tmp_g)]+min_lb_adjustment) / sum_lbs


        elif self.fairness_type == "stratified_g":
            for tmp_g in self.g_item:
                
                larger_loss_group = []
                smaller_loss_group = []
                
                for tmp_y in self.y_item:
                    if epoch_loss_yg[(tmp_y, tmp_g)] > epoch_loss_g[tmp_g]:
                        larger_loss_group.append(tmp_y)
                    elif epoch_loss_yg[(tmp_y, tmp_g)] < epoch_loss_g[tmp_g]:
                        smaller_loss_group.append(tmp_y)

                for llg in larger_loss_group:
                    self.lb_dict[(llg, tmp_g)] += self.alpha / len(larger_loss_group)
                for slg in smaller_loss_group:
                    self.lb_dict[(slg, tmp_g)] -= self.alpha / len(smaller_loss_group)

                # Normalize probabilities
                g_lbs = [self.lb_dict[(_y, tmp_g)] for _y in self.y_item]
                min_lb_adjustment = -1 * min([min(g_lbs), 0])
                sum_lbs = (sum(g_lbs) + min_lb_adjustment * len(self.y_item) ) / (self.g_len[tmp_g]/self.N)
                for tmp_y in self.y_item:
                    self.lb_dict[(tmp_y, tmp_g)] = (self.lb_dict[(tmp_y, tmp_g)]+min_lb_adjustment) / sum_lbs