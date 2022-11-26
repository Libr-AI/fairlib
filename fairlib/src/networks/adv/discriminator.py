import torch.nn as nn
import numpy as np
import torch
import logging
import time
from pathlib import Path
from .utils import SubDiscriminator
from .customized_loss import DiffLoss

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Train the discriminator 1 batch
def adv_train_batch(model, discriminators, batch, args):
    """train the discriminator one batch

    Args:
        model (torch.nn.Module): the main task model
        discriminators (torch.nn.Module): the discriminator
        batch (tuple): bach data, including inputs, target labels, protected labels, etc.
        args (namespace): arguments for training

    Returns:
        float: training loss
    """

    batch_loss = 0

    text = batch[0]
    tags = batch[1].long()
    p_tags = batch[2].float()
    adv_instance_weights = batch[4].float()

    # Remove instances that are not annotated with protected labels.
    if args.adv_decoupling:
        decoupling_masks = (adv_instance_weights != -1)
        text = text[decoupling_masks]
        tags = tags[decoupling_masks]
        p_tags = p_tags[decoupling_masks]
        adv_instance_weights = adv_instance_weights[decoupling_masks]

    text = text.to(args.device)
    tags = tags.to(args.device)
    p_tags = p_tags.to(args.device)
    adv_instance_weights = adv_instance_weights.to(args.device)

    if args.encoder_architecture == "BERT":
        # Modify the inputs for BERT models
        attention_mask = torch.stack(batch["attention_mask"]).float().squeeze().T
        if args.adv_decoupling:
            attention_mask = attention_mask[decoupling_masks]
        attention_mask = attention_mask.to(args.device)
        text = (text, attention_mask)
        
    # hidden representations from the model
    if args.gated:
        hs = model.hidden(text, p_tags).detach()
    else:
        hs = model.hidden(text).detach()

    # iterate all discriminators
    for discriminator in discriminators:

        adv_optimizer = discriminator.optimizer
        adv_criterion = discriminator.criterion

        adv_optimizer.zero_grad()

        if args.adv_gated:
            adv_predictions = discriminator(hs, tags.long())
        else:
            adv_predictions = discriminator(hs)

        # add the weighted loss
        if args.adv_BT is not None and args.adv_BT == "Reweighting":
            loss = adv_criterion(adv_predictions, p_tags.long())
            loss = torch.mean(loss * adv_instance_weights)
        else:
            loss = adv_criterion(adv_predictions, p_tags.long())

        # encrouge orthogonality
        if args.adv_num_subDiscriminator>1 and args.adv_diverse_lambda>0:
            # Get hidden representation.
            adv_hs_current = discriminator.hidden(hs, tags)
            for discriminator2 in discriminators:
                if discriminator != discriminator2:
                    adv_hs = discriminator2.hidden(hs, tags)
                    # Calculate diff_loss
                    # should not include the current model
                    difference_loss = args.adv_diverse_lambda * args.diff_loss(adv_hs_current, adv_hs)
                    loss = loss + difference_loss

        loss.backward()
        adv_optimizer.step()
        adv_optimizer.zero_grad()
        batch_loss += loss.item()
    return batch_loss

# train the discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, args):
    """train the discriminator one epoch

    Args:
        model (torch.nn.Module): the main task model.
        discriminators (torch.nn.Module): the discriminator
        iterator (dataloader): torch data iterator.
        args (namespace): arguments for training.

    Returns:
        float: training loss.
    """

    epoch_loss = 0
    model.eval()
    
    for discriminator in discriminators:
        discriminator.train()
    
    for it, batch in enumerate(iterator):

        batch_loss = adv_train_batch(model, discriminators, batch, args)

        epoch_loss += batch_loss

    return epoch_loss / len(iterator)

def adv_eval_epoch(model, discriminators, iterator, args):
    """evaluate the discriminator

    Args:
        model (torch.nn.Module): the main task model.
        discriminators (torch.nn.Module): the discriminator
        iterator (dataloader): torch data iterator.
        args (namespace): arguments for training.

    Returns:
        tuple: (evaluation loss, evaluation metrics)
    """

    epoch_loss = 0
    model.eval()
    
    for discriminator in discriminators:
        discriminator.eval()

    preds = [[] for i in range(args.adv_num_subDiscriminator)]
    labels = iterator.dataset.y
    private_labels = iterator.dataset.protected_label
    
    with torch.no_grad():
        for it, batch in enumerate(iterator):

            text = batch[0]
            tags = batch[1].long()
            p_tags = batch[2].float()

            if args.adv_BT is not None and args.adv_BT == "Reweighting":
                adv_instance_weights = batch[3].float()
                adv_instance_weights = adv_instance_weights.to(args.device)

            text = text.to(args.device)
            tags = tags.to(args.device)
            p_tags = p_tags.to(args.device)

            # hidden representations from the model
            if args.gated:
                hs = model.hidden(text, p_tags).detach()
            else:
                hs = model.hidden(text).detach()

            # iterate all discriminators
            for index, discriminator in enumerate(discriminators):

                adv_criterion = discriminator.criterion

                if args.adv_gated:
                    adv_predictions = discriminator(hs, tags.long())
                else:
                    adv_predictions = discriminator(hs)

                # add the weighted loss
                if args.adv_BT is not None and args.adv_BT == "Reweighting":
                    loss = adv_criterion(adv_predictions, p_tags.long())
                    loss = torch.mean(loss * adv_instance_weights)
                else:
                    loss = adv_criterion(adv_predictions, p_tags.long())

                # encrouge orthogonality
                if args.adv_num_subDiscriminator>1 and args.adv_diverse_lambda>0:
                    # Get hidden representation.
                    adv_hs_current = discriminator.hidden(hs, tags)
                    for discriminator2 in discriminators:
                        if discriminator != discriminator2:
                            adv_hs = discriminator2.hidden(hs, tags)
                            # Calculate diff_loss
                            # should not include the current model
                            difference_loss = args.adv_diverse_lambda * args.diff_loss(adv_hs_current, adv_hs)
                            loss = loss + difference_loss
                
                adv_predictions = adv_predictions.detach().cpu()
                preds[index] += list(torch.argmax(adv_predictions, axis=1).numpy())

                epoch_loss += loss.item()
    
    # Calculate the evaluation scores
    accuracy = []
    macro_fscore = []
    micro_fscore = []

    y_true = np.array(private_labels)
    for i in range(args.adv_num_subDiscriminator):
        y_pred = np.array(preds[i])
        accuracy.append(accuracy_score(y_true, y_pred))
        macro_fscore.append(f1_score(y_true, y_pred, average="macro"))
        micro_fscore.append(f1_score(y_true, y_pred, average="micro"))

    results_dic = {
        "loss" : (epoch_loss / len(iterator)/args.adv_num_subDiscriminator),
        "accuracy" : np.mean(accuracy),
        "macro_fscore" : np.mean(macro_fscore),
        "micro_fscore" : np.mean(micro_fscore)
    }

    return ((epoch_loss / len(iterator)), results_dic)

class Discriminator():
    def __init__(self, args):
        self.args = args

        # Init subdiscriminators
        self.subdiscriminators = []

        for _ in range(self.args.adv_num_subDiscriminator):
            self.subdiscriminators.append(SubDiscriminator(self.args))

        # Init difference loss for the diverse Adv
        if self.args.adv_num_subDiscriminator>1 and self.args.adv_diverse_lambda>0:
            self.diff_loss = DiffLoss()

        self.adv_loss_criterion = torch.nn.CrossEntropyLoss()
    
    
    def train_self_batch(self, model, batch):
        adv_train_batch(
            model = model, 
            discriminators = self.subdiscriminators, 
            batch = batch, 
            args = self.args)


    def train_self(self, model):
        # Use different datasets if decoupling
        if self.args.adv_decoupling:
            assert self.args.train_generator is not None, "A different train generator is required"

            self.train_iterator = self.args.opt.adv_train_generator
            self.dev_iterator = self.args.opt.adv_dev_generator
            self.test_iterator = self.args.opt.adv_test_generator
        else:
            self.train_iterator = self.args.opt.train_generator
            self.dev_iterator = self.args.opt.dev_generator
            self.test_iterator = self.args.opt.test_generator

        epochs_since_improvement = 0
        best_valid_loss = 1e+5
        
        self.eval_scores(model, additional_info = "Adv Evaluation before training")

        for epoch in range(self.args.opt.adv_epochs):
            
            # Early stopping
            if epochs_since_improvement >= self.args.adv_epochs_since_improvement:
                break
            
            # One epoch's training
            epoch_train_loss = adv_train_epoch(
                model = model, 
                discriminators = self.subdiscriminators, 
                iterator = self.train_iterator, 
                args = self.args)

            # One epoch's validation
            epoch_valid_loss, valid_results_dic = adv_eval_epoch(
                model = model, 
                discriminators = self.subdiscriminators, 
                iterator = self.dev_iterator, 
                args = self.args)

            # Check if there was an improvement
            is_best = epoch_valid_loss < best_valid_loss
            best_valid_loss = min(epoch_valid_loss, best_valid_loss)

            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0

                discriminator_state_dct = {j : self.subdiscriminators[j].state_dict() for j in range(self.args.adv_num_subDiscriminator)}
                torch.save(discriminator_state_dct, Path(self.args.model_dir) / "BEST_Discriminator.pth.tar")

        self.eval_scores(model, additional_info = "Evaluation at Adv Epoch %d" % (epoch,))

        # Reload parameters from the the best checkpoint
        best_checkpoint = torch.load(Path(self.args.model_dir) / "BEST_Discriminator.pth.tar", map_location=self.args.device)
        for j in range(self.args.adv_num_subDiscriminator):
            self.subdiscriminators[j].load_state_dict(best_checkpoint[j])
    
    def eval_scores(self,model, additional_info = None):
        # Output at the end
        # One epoch's validation
        epoch_valid_loss, valid_results_dic = adv_eval_epoch(
        model = model, 
        discriminators = self.subdiscriminators, 
        iterator = self.dev_iterator, 
        args = self.args)

        (epoch_test_loss, test_results_dic) = adv_eval_epoch(
            model = model, 
            discriminators = self.subdiscriminators, 
            iterator = self.test_iterator, 
            args = self.args)
        
        if additional_info is not None:
            logging.info(additional_info)

        logging.info(
            ('Validation Loss: {:2.2f} \tAcc: {:2.2f} \tMacroF1: {:2.2f} \tMicroF1: {:2.2f} \t').format(
                    valid_results_dic["loss"], 100. * valid_results_dic["accuracy"], 
                    100. * valid_results_dic["macro_fscore"], 100. * valid_results_dic["micro_fscore"]
                ))
        logging.info(
            ('Test Loss: {:2.2f} \tAcc: {:2.2f} \tMacroF1: {:2.2f} \tMicroF1: {:2.2f} \t').format(
                    test_results_dic["loss"], 100. * test_results_dic["accuracy"], 
                    100. * test_results_dic["macro_fscore"], 100. * test_results_dic["micro_fscore"]
                ))

    def adv_loss(self, hs, tags, p_tags):
        p_tags = p_tags.long()

        adv_losses = []
        for j in range(self.args.adv_num_subDiscriminator):
            if self.args.adv_gated:
                _adv_preds = self.subdiscriminators[j](hs, tags)
            else:
                _adv_preds = self.subdiscriminators[j](hs)

            if self.args.adv_uniform_label:
                # uniform labels
                batch_size, num_g_class = _adv_preds.shape
                # init uniform protected attributes
                p_tags = (1/num_g_class) * torch.ones_like(_adv_preds)
                p_tags = p_tags.to(self.args.device)
                # calculate the adv loss with the uniform protected labels
                # cross entropy loss for soft labels
            
            if self.args.adv_uniform_label:
                # Learn to predict uniform labels.
                adv_losses.append(-1*self.args.adv_lambda*self.adv_loss_criterion(_adv_preds, p_tags))
            else:
                # Unlearn to remove protected info.
                adv_losses.append(self.args.adv_lambda*self.adv_loss_criterion(_adv_preds, p_tags))

        return adv_losses