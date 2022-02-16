from re import S
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.modules import Module
from .utils import BaseModel


class MLP(BaseModel):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        
        assert args.n_hidden >= 0, "n_hidden must be nonnegative"
        self.hidden_layers = nn.ModuleList()

        if args.n_hidden == 0:
            self.output_layer = nn.Linear(args.emb_size, args.num_classes)
        else:
            self.input_layer = nn.Linear(args.emb_size, args.hidden_size)
            self.output_layer = nn.Linear(args.hidden_size, args.num_classes)

            # Init batch norm, dropout, and activation function
            self.init_hyperparameters()
            
            all_hidden_layers = [self.input_layer] + [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_hidden-1)]

            for _hidden_layer in all_hidden_layers:
                self.hidden_layers.append(_hidden_layer)
                if self.dropout is not None:
                    self.hidden_layers.append(self.dropout)
                if self.BN is not None:
                    self.hidden_layers.append(self.BN)
                if self.activation_function is not None:
                    self.hidden_layers.append(self.activation_function)

        self.init_for_training()

    def forward(self, input_data):
        for layer in self.hidden_layers:
            input_data = layer(input_data)
        output = self.output_layer(input_data)
        return output
    
    def hidden(self, input_data):
        assert self.args.adv_level in ["input", "last_hidden", "output"]

        if self.args.adv_level == "input":
            return input_data
        else:
            for layer in self.hidden_layers:
                input_data = layer(input_data)
            if self.args.adv_level == "last_hidden":
                return input_data
            elif self.args.adv_level == "output":
                output = self.output_layer(input_data)
                return output
            else:
                raise "not implemented yet"