from .debias import get_debiasing_projection, get_projection_to_intersection_of_nullspaces

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from collections import Counter, defaultdict

import warnings
warnings.filterwarnings('ignore')

import torch
from pathlib import Path
import numpy as np

from ...evaluators import present_evaluation_scores
import logging


def save_INLP_checkpoint(
    epoch, dev_predictions, test_predictions, dev_evaluations, test_evaluations, 
    checkpoint_dir,dev_confusion_matrices, test_confusion_matrices):

    _state = {
        'epoch': epoch,
        # 'dev_predictions': dev_predictions,
        # 'test_predictions': test_predictions,
        'dev_evaluations': dev_evaluations,
        'test_evaluations': test_evaluations,
        "valid_confusion_matrices" : dev_confusion_matrices,
        "test_confusion_matrices" : test_confusion_matrices,
        }

    filename = 'INLP_checkpoint_' + "iteration{}".format(epoch) + '.pth.tar'
    torch.save(_state, Path(checkpoint_dir) / filename)

def load_trained_model(model, checkpoint_dir, device):
    checkpoint_PATH = Path(checkpoint_dir) / 'BEST_checkpoint.pth.tar'
    model.load_state_dict(torch.load(checkpoint_PATH)["model"])
    model.to(device)
    model.eval()
    return model

def extract_hidden_representations(model, split, args):
    hidden = []
    labels = []
    private_labels = []

    if split == "train":
        iterator = args.train_generator
    elif split == "dev":
        iterator = args.dev_generator
    elif split == "test":
        iterator = args.test_generator
    else:
        raise NotImplementedError

    for batch in iterator:
        
        text = batch[0].squeeze()
        tags = batch[1].squeeze()
        p_tags = batch[2].squeeze()
        
        labels += list(tags.cpu().numpy() )
        private_labels += list(p_tags.cpu().numpy())

        text = text.to(args.device)
        tags = tags.to(args.device).long()
        p_tags = p_tags.to(args.device).float()

        # main model predictions
        if args.gated:
            hidden_state = model.hidden(text, p_tags)
        else:
            hidden_state = model.hidden(text)
        
        hidden.append(hidden_state.detach().cpu().numpy())
    
    hidden = np.concatenate(hidden, 0)

    hidden = np.array(hidden)
    labels = np.array(labels)
    private_labels = np.array(private_labels)

    return hidden, labels, private_labels

def get_INLP_trade_offs(model, args):
    # Hyperparameters
    discriminator_reweighting = args.INLP_discriminator_reweighting
    by_class = args.INLP_by_class
    n = args.INLP_n
    min_acc = args.INLP_min_acc
    is_autoregressive = True
    dim = args.hidden_size
    clf = LogisticRegression
    clf_params = {'fit_intercept': True, 'class_weight': discriminator_reweighting, 'dual': False, 'C': 0.1, "max_iter": 100}

    # Load best checkpoints
    model = load_trained_model(model, args.model_dir, args.device)

    # Extract Hidden representations
    train_hidden, train_labels, train_private_labels = extract_hidden_representations(model, "train", args)
    dev_hidden, dev_labels, dev_private_labels = extract_hidden_representations(model, "dev", args)
    test_hidden, test_labels, test_private_labels = extract_hidden_representations(model, "test", args)

    # Run INLP

    P_n = get_debiasing_projection(clf, clf_params, n, dim, is_autoregressive, min_acc,
                                    train_hidden, train_private_labels, dev_hidden, dev_private_labels,
                                    by_class=by_class, Y_train_main=train_labels, Y_dev_main=dev_labels)
    
    rowspaces = P_n[1]

    for iteration, p_iteration in enumerate(range(1, len(rowspaces))):
        
        P = get_projection_to_intersection_of_nullspaces(rowspaces[:p_iteration], input_dim=300)
        
        debiased_x_train = P.dot(train_hidden.T).T
        debiased_x_dev = P.dot(dev_hidden.T).T
        debiased_x_test = P.dot(test_hidden.T).T
        
        classifier = LogisticRegression(warm_start = True, 
                                            penalty = 'l2',
                                            solver = "sag", 
                                            multi_class = 'multinomial', 
                                            fit_intercept = True,
                                            verbose = 0, 
                                            max_iter = 10,
                                            n_jobs = 24, 
                                            random_state = 1)

        classifier.fit(debiased_x_train, train_labels)
        
        # Evaluation
        dev_y_pred = classifier.predict(debiased_x_dev)
        test_y_pred= classifier.predict(debiased_x_test)

        logging.info("Evaluation at Epoch %d" % (iteration,))

        present_evaluation_scores(
            valid_preds = dev_y_pred, valid_labels = dev_labels, 
            valid_private_labels = dev_private_labels,
            test_preds = test_y_pred, test_labels = test_labels, 
            test_private_labels = test_private_labels,
            epoch = iteration, epochs_since_improvement = None, 
            model = model, epoch_valid_loss = None,
            is_best = False, prefix = "INLP_checkpoint",
            )

    return None
