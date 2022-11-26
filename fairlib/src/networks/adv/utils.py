import torch.nn as nn
import numpy as np
import torch
import logging
from torch.optim import Adam
import time
from pathlib import Path
from ..augmentation_layer import Augmentation_layer

class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


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
        if self.args.adv_n_hidden == 0:
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
        self.grad_rev = GradientReversal(self.args.adv_lambda)
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

        # Add the onehot target to the input of adv
        if self.args.adv_gated and self.args.adv_gated_type == "Inputs":
            self.input_dim = self.input_dim + self.args.num_classes
            # One-hot mapping for the class
            self.mapping = torch.eye(self.args.num_classes, requires_grad=False)
            self.mapping = self.mapping.to(self.args.device)

        if args.adv_n_hidden == 0:
            self.output_layer = nn.Linear(self.input_dim, args.num_groups)

        else:
            self.output_layer = nn.Linear(args.adv_hidden_size, args.num_groups)
        
        # Init batch norm, dropout, and activation function
        self.init_hyperparameters()
        # Hidden layers
        self.hidden_layers = self.init_hidden_layers()
            
        # Augmentation layers
        if self.args.adv_gated and self.args.adv_gated_type == "Augmentation":
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
                self.augmentation_components = Augmentation_layer(
                    mapping=self.mapping,
                    num_component=self.args.num_classes,
                    device=self.args.device,
                    sample_component=self.hidden_layers
                )

        self.init_for_training()

    def forward(self, input_data, group_label = None):
        if (self.args.adv_gated):
            assert group_label is not None, "Group labels are needed for augmentation"
            
            if (self.args.adv_gated_type == "Inputs"):
                # Get one hot representations of y
                onehot_y = self.mapping[group_label.long()]
                # Concat 
                input_data = torch.cat([input_data, onehot_y], dim=1)
        
        # Main model
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if (self.args.adv_gated) and (
            self.args.adv_n_hidden > 0) and (
            self.args.adv_gated_type == "Augmentation"):

            specific_output = self.augmentation_components(input_data, group_label)

            main_output = main_output + specific_output

        output = self.output_layer(main_output)
        return output
    
    def hidden(self, input_data, group_label = None):
        if (self.args.adv_gated):
            assert group_label is not None, "Group labels are needed for augmentation"
            
            if (self.args.adv_gated_type == "Inputs"):
                # Get one hot representations of y
                onehot_y = self.mapping[group_label.long()]
                # Concat 
                input_data = torch.cat([input_data, onehot_y], dim=1)
        
        # Main model
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if (self.args.adv_gated) and (
            self.args.adv_n_hidden > 0) and (
            self.args.adv_gated_type == "Augmentation"):

            specific_output = self.augmentation_components(input_data, group_label)

            main_output = main_output + specific_output
        
        return main_output