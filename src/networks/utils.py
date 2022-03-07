import torch.nn as nn
import torch
import logging
from torch.optim import Adam
import time
from pathlib import Path
# from .evaluator import gap_eval_scores
from ..evaluators.evaluator import gap_eval_scores

def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def save_checkpoint(
    epoch, epochs_since_improvement, model, loss, 
    dev_predictions, test_predictions, dev_evaluations, 
    test_evaluations, is_best, checkpoint_dir):

    _state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        # 'model': model.state_dict(),
        'loss': loss,
        'dev_predictions': dev_predictions,
        'test_predictions': test_predictions,
        'dev_evaluations': dev_evaluations,
        'test_evaluations': test_evaluations
        }

    filename = 'checkpoint_' + "epoch{}".format(epoch) + '.pth.tar'
    torch.save(_state, Path(checkpoint_dir) / filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        _state["model"]=model.state_dict()
        torch.save(_state, Path(checkpoint_dir) / 'BEST_checkpoint.pth.tar')

# train the main model with adv loss
def train_epoch(model, iterator, args, epoch):

    epoch_loss = 0
    model.train()

    optimizer = model.optimizer
    criterion = model.criterion

    data_t0 = time.time()
    data_t, t = 0, 0
    
    for it, batch in enumerate(iterator):
        
        text = batch[0].squeeze()
        tags = batch[1].long().squeeze()
        p_tags = batch[2].float().squeeze()

        if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(args.device)

        text = text.to(args.device)
        tags = tags.to(args.device)
        p_tags = p_tags.to(args.device)

        data_t += (time.time() - data_t0)
        t0 = time.time()
        
        optimizer.zero_grad()
        # main model predictions
        if args.gated:
            predictions = model(text, p_tags)
        else:
            predictions = model(text)
        # main tasks loss
        # add the weighted loss
        if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags)
            loss = torch.mean(loss * instance_weights)
        else:
            loss = criterion(predictions, tags)

        if args.adv_debiasing:
            # Update discriminator if needed
            if args.adv_update_frequency == "Batch":
                args.discriminator.train_self_batch(model, batch)

            # get hidden representations
            if args.gated:
                hs = model.hidden(text, p_tags)
            else:
                hs = model.hidden(text)

            adv_losses = args.discriminator.adv_loss(hs, tags, p_tags)

            for adv_loss in adv_losses:
                loss = loss - (adv_loss / args.adv_num_subDiscriminator)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        t += (time.time() - t0)
        data_t0 = time.time()

        if it % args.log_interval == 0:
            logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * args.batch_size, len(iterator.dataset),
                    100. * it / len(iterator), loss, data_t, t,
                ))
            data_t, t = 0, 0
        
    return epoch_loss / len(iterator)


# to evaluate the main model
def eval_epoch(model, iterator, args):
    
    epoch_loss = 0
    device = args.device
    
    model.eval()

    criterion = model.criterion

    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        p_tags = batch[2]

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).float()

        if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(device)

        # main model predictions
        if args.gated:
            predictions = model(text, p_tags)
        else:
            predictions = model(text)
        
        # add the weighted loss
        if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags)
            loss = torch.mean(loss * instance_weights)
        else:
            loss = criterion(predictions, tags)
                        
        epoch_loss += loss.item()
        
        predictions = predictions.detach().cpu()
        tags = tags.cpu().numpy()

        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

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
    
    def train_self(self):
        
        # Reinit the train loader for DyBT
        if self.args.DyBT is not None and self.args.DyBT == 'FairBatch':
            from .fairbatch import FairBatch
            # Init the fairbatch sampler
            fairbatch_sampler = FairBatch(self, self.args)
            # Replace the tran iterator with fairbatch version
            self.args.opt.train_generator = torch.utils.data.DataLoader(self.args.opt.train_generator.dataset, sampler=fairbatch_sampler, num_workers=0)

        epochs_since_improvement = 0
        best_valid_loss = 1e+5

        for epoch in range(self.args.opt.epochs):
            
            # Early stopping
            if epochs_since_improvement >= self.args.epochs_since_improvement:
                break
            
            # One epoch's training
            epoch_train_loss = train_epoch(
                model = self, 
                iterator = self.args.opt.train_generator, 
                args = self.args, 
                epoch = epoch)

            # One epoch's validation
            (epoch_valid_loss, valid_preds, 
            valid_labels, valid_private_labels) = eval_epoch(
                model = self, 
                iterator = self.args.opt.dev_generator, 
                args = self.args)

            # Update discriminator if needed
            if self.args.adv_debiasing and self.args.adv_update_frequency == "Epoch":
                self.args.discriminator.train_self(self)

            # Check if there was an improvement
            is_best = epoch_valid_loss < best_valid_loss
            best_valid_loss = min(epoch_valid_loss, best_valid_loss)

            if not is_best:
                epochs_since_improvement += 1
                logging.info("Epochs since last improvement: %d" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            if epoch % self.args.checkpoint_interval == 0:
                valid_scores = gap_eval_scores(
                    y_pred=valid_preds,
                    y_true=valid_labels, 
                    protected_attribute=valid_private_labels)

                (epoch_test_loss, test_preds, 
                test_labels, test_private_labels) = eval_epoch(
                    model = self, 
                    iterator = self.args.opt.test_generator, 
                    args = self.args)
                
                test_scores = gap_eval_scores(
                    y_pred=test_preds,
                    y_true=test_labels, 
                    protected_attribute=test_private_labels)

                # Save checkpoint
                save_checkpoint(
                    epoch = epoch, 
                    epochs_since_improvement = epochs_since_improvement, 
                    model = self, 
                    loss = epoch_valid_loss, 
                    dev_predictions = valid_preds, 
                    test_predictions = test_preds,
                    dev_evaluations = valid_scores, 
                    test_evaluations = test_scores,
                    is_best = is_best,
                    checkpoint_dir = self.args.model_dir)
                
                logging.info("Evaluation at Epoch %d" % (epoch,))
                logging.info((
                    'Validation GAP: {:2.2f}\tAcc: {:2.2f}\tMacroF1: {:2.2f}\tMicroF1: {:2.2f}'
                ).format(
                    100. * valid_scores["rms_TPR"], 100. * valid_scores["accuracy"], 
                    100. * valid_scores["macro_fscore"], 100. * valid_scores["micro_fscore"]
                ))
                logging.info((
                    'Test GAP: {:2.2f}\tAcc: {:2.2f}\tMacroF1: {:2.2f}\tMicroF1: {:2.2f}'
                ).format(
                    100. * test_scores["rms_TPR"], 100. * test_scores["accuracy"], 
                    100. * test_scores["macro_fscore"], 100. * test_scores["micro_fscore"]
                ))