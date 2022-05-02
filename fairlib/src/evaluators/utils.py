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


def save_checkpoint(
    epoch, epochs_since_improvement, model, loss, dev_evaluations,
    valid_confusion_matrices, test_confusion_matrices,
    test_evaluations, is_best, checkpoint_dir, dev_predictions=None, test_predictions=None):

    _state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        # 'model': model.state_dict(),
        'loss': loss,
        # 'dev_predictions': dev_predictions,
        # 'test_predictions': test_predictions,
        "valid_confusion_matrices" : valid_confusion_matrices,
        "test_confusion_matrices" : test_confusion_matrices,
        'dev_evaluations': dev_evaluations,
        'test_evaluations': test_evaluations
        }

    if dev_predictions is not None:
        _state["dev_predictions"] = dev_predictions
    if test_predictions is not None:
        _state["test_predictions"] = test_predictions

    filename = 'checkpoint_' + "epoch{:.2f}".format(epoch) + '.pth.tar'
    torch.save(_state, Path(checkpoint_dir) / filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        _state["model"]=model.state_dict()
        torch.save(_state, Path(checkpoint_dir) / 'BEST_checkpoint.pth.tar')