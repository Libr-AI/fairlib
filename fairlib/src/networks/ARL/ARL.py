import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from ..adv.utils import BaseDiscriminator
from ..augmentation_layer import Augmentation_layer



class ARL(BaseDiscriminator):
    def __init__(self, args):
        """The class for ARL training
        This class is similar to the standard adversary for adversarial training, so it is 
        implemented as a child class of the BaseDiscriminator, which is the same the other
        sub-discriminators. Main differences includes:
        1. the forward function of ARL returns the predicted weights of each instance.
        2. the ARL is trained to predict weights such that the weighted loss of the main model is maximized.


        Args:
            args (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        super(ARL, self).__init__()
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

        # Add the onehot target to the input of adv
        if self.args.adv_gated and self.args.adv_gated_type == "Inputs":
            self.input_dim = self.input_dim + self.args.num_classes
            # One-hot mapping for the class
            self.mapping = torch.eye(self.args.num_classes, requires_grad=False)
            self.mapping = self.mapping.to(self.args.device)

        # The output dim of the layer is 1, correponding to the weight for each instance.
        if args.adv_n_hidden == 0:
            self.output_layer = nn.Linear(self.input_dim, 1)

        else:
            self.output_layer = nn.Linear(args.adv_hidden_size, 1)

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
        self.sigmoid = nn.Sigmoid()

        if self.args.regression:
            self.criterion = torch.nn.MSELoss(reduction = "none")
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction = "none")

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

        main_output = self.output_layer(main_output)
        # Normalize outputs
        main_output = self.sigmoid(main_output)
        main_output = main_output / torch.mean(main_output)
        main_output = main_output + torch.ones_like(main_output)

        return main_output

    # Train the discriminator 1 batch
    def step(self, hs, preds, tags, args, train_step=True):
        """train the discriminator one batch

        Args:
            model (torch.nn.Module): the main task model
            batch (tuple): bach data, including inputs, target labels, protected labels, etc.
            args (namespace): arguments for training

        Returns:
            float: weights
        """

        if train_step:
            self.train()
        else:
            self.eval()

        if args.adv_gated:
            adv_predictions = self(hs, tags.long())
        else:
            adv_predictions = self(hs)

        if train_step:
            adv_optimizer = self.optimizer
            adv_optimizer.zero_grad()

            weighted_loss = self.get_adversary_loss(preds, tags, adv_predictions)
            # print(weighted_loss)
            weighted_loss.backward(retain_graph=True)
            # print(self.output_layer.weight.grad)
            adv_optimizer.step()
            adv_optimizer.zero_grad()

        return adv_predictions

    def get_learner_loss(self, preds, tags, adversarial_weights):
        """
        Compute the loss for the  main task model.
        """
        loss = self.criterion(preds, tags)
        weighted_loss = loss * adversarial_weights.squeeze()
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_adversary_loss(self, preds, tags, adversarial_weights):
        """
        Compute the loss for the adversary.
        """
        return -1 * self.get_learner_loss(preds, tags, adversarial_weights)

    def get_arl_loss(self, model, batch, predictions, args):
        """Get ARL loss

        Args:
            model (torch.nn.Module): the main task model
            batch (tuple): bach data, including inputs, target labels, protected labels, etc.
            args (namespace): arguments for training

        Returns:
            float: training loss
        """

        text = batch[0]
        tags = batch[1].long()

        text = text.to(args.device)
        tags = tags.to(args.device)

        hs = model.hidden(text).detach()
        preds = model(text).detach()

        for _ in range(args.ARL_n):
            # Train the adversary one step
            _ = self.step(hs, preds, tags, args, train_step=True)
        # Get the adversarial weights
        adversarial_weights = self.step(hs, preds, tags, args, train_step=False)
        # print(adversarial_weights)
        # Weighted loss
        arl_weighted_loss = self.get_learner_loss(predictions, tags, adversarial_weights)
        # Unweighted loss, i.e., the standard loss for the vanilla model
        # vanilla_unweighted_loss = self.get_learner_loss(predictions, tags, torch.ones_like(adversarial_weights))
        # By default, replace the vanilla loss with ARL
        # arl_loss = arl_weighted_loss - vanilla_unweighted_loss

        return arl_weighted_loss