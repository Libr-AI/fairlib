import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.modules import Module
import logging
from torch.optim import Adam

def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)

class BaseModel(nn.Module):

    def init_for_training(self):

        self.device = self.args.device
        self.to(self.device)

        self.learning_rate = self.args.lr
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        if self.args.BT and self.args.BT == "Reweighting":
            self.criterion = torch.nn.CrossEntropyLoss(reduction = "none")
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        print_network(self, verbose=True)

    def init_hyperparameters(self):
        if self.args.activation_function == "ReLu":
            self.AF = nn.ReLU()
        elif self.args.activation_function == "Tanh":
            self.AF = nn.Tanh()
        elif self.args.activation_function == "LeakyReLU":
            self.AF = nn.LeakyReLU()
        else:
            raise "not implemented yet"

        
        if self.args.batch_norm:
            self.BN = nn.BatchNorm1d(self.args.hidden_size)
        else:
            self.BN = None

        assert (self.args.dropout >= 0) and (self.args.dropout <= 1), "Probability must be in the range from 0 to 1"
        if self.args.dropout > 0:
            self.dropout = nn.Dropout(p=self.args.dropout)
        else:
            self.dropout = None