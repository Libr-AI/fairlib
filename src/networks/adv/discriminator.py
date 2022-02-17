import torch.nn as nn
import numpy as np
import torch
import logging
import time
from pathlib import Path
from .utils import SubDiscriminator
from .customized_loss import DiffLoss, cross_entropy_with_probs

# train the discriminator 1 epoch
def adv_train_epoch(model, discriminators, iterator, args):

    epoch_loss = 0
    model.eval()
    
    for discriminator in discriminators:
        discriminator.train()

    optimizer = model.optimizer
    criterion = model.criterion

    data_t0 = time.time()
    
    for it, batch in enumerate(iterator):

        data_t = time.time() - data_t0
        t0 = time.time()
        
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
        for discriminator in zip(discriminators):

            adv_optimizer = discriminator.optimizer
            adv_criterion = discriminator.criterion

            adv_optimizer.zero_grad()

            if args.adv_gated:
                adv_predictions = discriminator(hs, tags.long())
            else:
                adv_predictions = discriminator(hs)


            # add the weighted loss
            if args.adv_BT is not None and args.adv_BT == "Reweighting":
                loss = criterion(adv_predictions, p_tags)
                loss = torch.mean(loss * adv_instance_weights)
            else:
                loss = criterion(adv_predictions, p_tags)

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
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

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

        # if self.args.adv_uniform_label:
        # added to the adv loss function directly
