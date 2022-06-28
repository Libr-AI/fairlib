from .debias import get_debiasing_projection, get_projection_to_intersection_of_nullspaces

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from collections import Counter, defaultdict

import warnings
warnings.filterwarnings('ignore')

import torch
from pathlib import Path
import numpy as np

from ...evaluators import present_evaluation_scores
import logging

def load_trained_model(model, checkpoint_dir, device):
    checkpoint_PATH = Path(checkpoint_dir) / 'BEST_checkpoint.pth.tar'
    model.load_state_dict(torch.load(checkpoint_PATH)["model"])
    model.to(device)
    model.eval()
    return model

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
    train_hidden, train_labels, train_private_labels, train_regression_labels = model.extract_hidden_representations("train")
    dev_hidden, dev_labels, dev_private_labels, dev_regression_labels = model.extract_hidden_representations("dev")
    test_hidden, test_labels, test_private_labels, test_regression_labels = model.extract_hidden_representations("test")

    # Run INLP

    P_n = get_debiasing_projection(clf, clf_params, n, dim, is_autoregressive, min_acc,
                                    train_hidden, train_private_labels, dev_hidden, dev_private_labels,
                                    by_class=by_class, Y_train_main=train_labels, Y_dev_main=dev_labels)
    
    rowspaces = P_n[1]

    for iteration, p_iteration in enumerate(range(1, len(rowspaces))):
        
        P = get_projection_to_intersection_of_nullspaces(rowspaces[:p_iteration], input_dim=train_hidden.shape[1])
        
        debiased_x_train = P.dot(train_hidden.T).T
        debiased_x_dev = P.dot(dev_hidden.T).T
        debiased_x_test = P.dot(test_hidden.T).T
        
        if not args.regression:
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
        else:
            classifier = LinearRegression()
            classifier.fit(debiased_x_train, train_regression_labels)

        
        
        # Evaluation
        dev_y_pred = classifier.predict(debiased_x_dev)
        test_y_pred= classifier.predict(debiased_x_test)

        logging.info("Evaluation at Epoch %d" % (iteration,))

        present_evaluation_scores(
            valid_preds = dev_y_pred, 
            valid_labels = dev_labels if not args.regression else dev_regression_labels, 
            valid_private_labels = dev_private_labels,
            test_preds = test_y_pred, 
            test_labels = test_labels if not args.regression else test_regression_labels, 
            test_private_labels = test_private_labels,
            epoch = iteration, epochs_since_improvement = None, 
            model = model, epoch_valid_loss = None,
            is_best = False, prefix = "INLP_checkpoint",
            )

    return None
