import torch.nn as nn
import numpy as np
import torch
import logging
from torch.optim import Adam
import time
from pathlib import Path

def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)

class BaseDiscriminator(nn.Module):

    def init_for_training(self):

        self.device = self.args.device
        self.to(self.device)

        self.learning_rate = self.args.adv_lr
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        if self.args.adv_BT and self.args.adv_BT == "Reweighting":
            self.criterion = torch.nn.CrossEntropyLoss(reduction = "none")
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        print_network(self, verbose=True)

    def init_hyperparameters(self):
        if self.args.adv_activation_function == "ReLu":
            self.AF = nn.ReLU()
        elif self.args.adv_activation_function == "Tanh":
            self.AF = nn.Tanh()
        elif self.args.adv_activation_function == "LeakyReLU":
            self.AF = nn.LeakyReLU()
        else:
            raise "not implemented yet"

        if self.args.adv_batch_norm:
            self.BN = nn.BatchNorm1d(self.args.adv_hidden_size)
        else:
            self.BN = None

        assert (self.args.adv_dropout >= 0) and (self.args.adv_dropout <= 1), "Probability must be in the range from 0 to 1"
        if self.args.adv_dropout > 0:
            self.dropout = nn.Dropout(p=self.args.adv_dropout)
        else:
            self.dropout = None
    
    def init_hidden_layers(self):
        if self.adv_n_hidden == 0:
            return nn.ModuleList()
        else:
            # Hidden layers
            hidden_layers = nn.ModuleList()
            
            all_hidden_layers = [nn.Linear(self.input_dim, self.args.adv_hidden_size)] + [nn.Linear(self.args.adv_hidden_size, self.args.adv_hidden_size) for _ in range(self.args.adv_n_hidden-1)]

            for _hidden_layer in all_hidden_layers:
                hidden_layers.append(_hidden_layer)
                if self.dropout is not None:
                    hidden_layers.append(self.dropout)
                if self.BN is not None:
                    hidden_layers.append(self.BN)
                if self.AF is not None:
                    hidden_layers.append(self.AF)
            
            return hidden_layers


class SubDiscriminator(BaseDiscriminator):
    def __init__(self, args):
        super(SubDiscriminator, self).__init__()
        self.args = args
        
        assert args.adv_n_hidden >= 0, "n_hidden must be nonnegative"
        
        assert self.args.adv_level in ["input", "last_hidden", "output"]
        if self.args.adv_level == "input":
            self.input_dim = self.args.emb_size
        elif self.args.adv_level == "last_hidden":
            if self.args.n_hidden == 0:
                self.input_dim = self.args.emb_size
            else:
                self.input_dim = self.args.hidden_size
        elif self.args.adv_level == "output":
            self.input_dim = self.args.num_classes
        else:
            pass

        if args.adv_n_hidden == 0:
            self.output_layer = nn.Linear(self.input_dim, args.adv_num_classes)

        else:
            self.output_layer = nn.Linear(args.adv_hidden_size, args.adv_num_classes)
        
        # Init batch norm, dropout, and activation function
        self.init_hyperparameters()
        # Hidden layers
        self.hidden_layers = self.init_hidden_layers()
            
        # Augmentation layers
        if self.args.adv_gated:
            if self.args.adv_n_hidden == 0:
                logging.info("Gated component requires at least one hidden layers in the model")
                pass
            else:
                # Init the mapping for the augmentation layer
                if self.args.adv_gated_mapping is None:
                    # For each class init a discriminator component
                    self.mapping = torch.eye(self.args.num_classes, requires_grad=False)
                else:
                    # self.mapping = torch.from_numpy(mapping, requires_grad=False)
                    raise NotImplementedError

                # Init the augmentation layer
                self.augmentation_components = nn.ModuleList()
                for _ in range(self.args.num_classes):
                    # Assuming that the augmentation layers share the same architecture to the normal hidden layers
                    self.augmentation_components.append(self.init_hidden_layers())

        self.init_for_training()

    def forward(self, input_data, group_label = None):
        
        # Main model
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if self.args.adv_gated and self.args.adv_n_hidden > 0:
            assert group_label is not None, "Group labels are needed for augmentaiton"

            specific_output = []
            # Get group-specific representations
            # number_classes * batch_size * adv_units
            for _group_id in range(self.args.num_classes):
                _group_output = input_data
                for layer in self.augmentation_components[_group_id]:
                    _group_output = layer(_group_output)
                specific_output.append(_group_output) # batch_size * adv_units
            
            # Reshape the out_g to batch*num_classes*adv_units
            specific_output = [i.unsqueeze(dim=1) for i in specific_output] # Each element has the shape: batch_size * 1 * adv_units
            specific_output = torch.cat(specific_output, dim=1)

            # Mapping the group label to one-hot representation
            group_label = self.mapping[group_label.long()] # batch_size * num_classes
            group_label = group_label.unsqueeze(dim=1) # batch_size * 1 * num_classes

            # (batch_size * 1 * num_classes) * (batch*num_classes*adv_units)
            specific_output = torch.matmul(group_label.to(self.device), specific_output) # (batch_size * 1 * adv_units)
            specific_output = specific_output.squeeze(dim=1) # (batch_size * adv_units)

            main_output = main_output + specific_output

        output = self.output_layer(main_output)
        return output
    
    def hidden(self, input_data, group_label = None):
        # Main model
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if self.args.adv_gated and self.args.adv_n_hidden > 0:
            assert group_label is not None, "Group labels are needed for augmentaiton"

            specific_output = []
            # Get group-specific representations
            # number_classes * batch_size * adv_units
            for _group_id in range(self.args.num_classes):
                _group_output = input_data
                for layer in self.augmentation_components[_group_id]:
                    _group_output = layer(_group_output)
                specific_output.append(_group_output) # batch_size * adv_units
            
            # Reshape the out_g to batch*num_classes*adv_units
            specific_output = [i.unsqueeze(dim=1) for i in specific_output] # Each element has the shape: batch_size * 1 * adv_units
            specific_output = torch.cat(specific_output, dim=1)

            # Mapping the group label to one-hot representation
            group_label = self.mapping[group_label.long()] # batch_size * num_classes
            group_label = group_label.unsqueeze(dim=1) # batch_size * 1 * num_classes

            # (batch_size * 1 * num_classes) * (batch*num_classes*adv_units)
            specific_output = torch.matmul(group_label.to(self.device), specific_output) # (batch_size * 1 * adv_units)
            specific_output = specific_output.squeeze(dim=1) # (batch_size * adv_units)

            main_output = main_output + specific_output
        
        return main_output