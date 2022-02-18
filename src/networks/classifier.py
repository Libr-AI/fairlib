from re import S
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.modules import Module
from .utils import BaseModel
import logging


class MLP(BaseModel):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        
        assert args.n_hidden >= 0, "n_hidden must be nonnegative"
        if args.n_hidden == 0:
            self.output_layer = nn.Linear(args.emb_size, args.num_classes)
        else:
            self.output_layer = nn.Linear(args.hidden_size, args.num_classes)

        # Init batch norm, dropout, and activation function
        self.init_hyperparameters()
        
        # Init hidden layers
        self.hidden_layers = self.init_hidden_layers()

        # Augmentation layers
        if self.args.gated:
            if self.args.n_hidden == 0:
                logging.info("Gated component requires at least one hidden layers in the model")
                pass
            else:
                # Init the mapping for the augmentation layer
                if self.args.gated_mapping is None:
                    # For each class init a discriminator component
                    self.mapping = torch.eye(self.args.adv_num_classes, requires_grad=False)
                else:
                    # self.mapping = torch.from_numpy(mapping, requires_grad=False)
                    raise NotImplementedError

                # Init the augmentation layer
                self.augmentation_components = nn.ModuleList()
                for _ in range(self.args.adv_num_classes):
                    # Assuming that the augmentation layers share the same architecture to the normal hidden layers
                    self.augmentation_components.append(self.init_hidden_layers())
        
        self.init_for_training()

    def forward(self, input_data, group_label = None):
        # main out
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if self.args.gated and self.args.n_hidden > 0:
            assert group_label is not None, "Group labels are needed for augmentaiton"

            specific_output = []
            # Get group-specific representations
            # number_classes * batch_size * adv_units
            for _group_id in range(self.args.adv_num_classes):
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
        assert self.args.adv_level in ["input", "last_hidden", "output"]

        if self.args.adv_level == "input":
            return input_data
        else:
            # main out
            main_output = input_data
            for layer in self.hidden_layers:
                main_output = layer(main_output)

            # Augmentation
            if self.args.gated and self.args.n_hidden > 0:
                assert group_label is not None, "Group labels are needed for augmentaiton"

                specific_output = []
                # Get group-specific representations
                # number_classes * batch_size * adv_units
                for _group_id in range(self.args.adv_num_classes):
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
            if self.args.adv_level == "last_hidden":
                return main_output
            elif self.args.adv_level == "output":
                output = self.output_layer(main_output)
                return output
            else:
                raise "not implemented yet"
    
    def init_hidden_layers(self):
        args = self.args

        if args.n_hidden == 0:
            return nn.ModuleList()
        else:
            hidden_layers = nn.ModuleList()
            
            all_hidden_layers = [nn.Linear(args.emb_size, args.hidden_size)] + [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_hidden-1)]

            for _hidden_layer in all_hidden_layers:
                hidden_layers.append(_hidden_layer)
                if self.dropout is not None:
                    hidden_layers.append(self.dropout)
                if self.BN is not None:
                    hidden_layers.append(self.BN)
                if self.AF is not None:
                    hidden_layers.append(self.AF)
            return hidden_layers