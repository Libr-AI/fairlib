from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from collections import defaultdict, Counter


def leakage_evaluation(model, 
                    adv_level, 
                    training_generator,
                    validation_generator,
                    test_generator,
                    device,
                    augmentation = False):
    model.eval()
    model.adv_level = adv_level

    train_hidden = []
    train_labels = []
    train_private_labels = []

    for batch in training_generator:
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        train_labels += list(tags.cpu().numpy() )
        train_private_labels += list(p_tags.cpu().numpy())
        
        text = text.to(device)
        p_tags = p_tags.to(device)

        # hidden_state = model.hidden(text)
        if augmentation:
            hidden_state = model.hidden(text, p_tags)
        else:
            hidden_state = model.hidden(text)
        train_hidden.append(hidden_state.detach().cpu().numpy())
    train_hidden = np.concatenate(train_hidden, 0)

    dev_hidden = []
    dev_labels = []
    dev_private_labels = []

    for batch in validation_generator:
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        dev_labels += list(tags.cpu().numpy() )
        dev_private_labels += list(p_tags.cpu().numpy())

        text = text.to(device)
        p_tags = p_tags.to(device)

        # hidden_state = model.hidden(text)
        if augmentation:
            hidden_state = model.hidden(text, p_tags)
        else:
            hidden_state = model.hidden(text)
        dev_hidden.append(hidden_state.detach().cpu().numpy())
    dev_hidden = np.concatenate(dev_hidden, 0)

    test_hidden = []
    test_labels = []
    test_private_labels = []

    for batch in test_generator:
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        test_labels += list(tags.cpu().numpy() )
        test_private_labels += list(p_tags.cpu().numpy())
        
        text = text.to(device)
        p_tags = p_tags.to(device)

        # hidden_state = model.hidden(text)
        if augmentation:
            hidden_state = model.hidden(text, p_tags)
        else:
            hidden_state = model.hidden(text)
        test_hidden.append(hidden_state.detach().cpu().numpy())
    test_hidden = np.concatenate(test_hidden, 0)


    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    # biased_classifier = MLPClassifier(max_iter=50, batch_size=1024)
    # biased_classifier.fit(train_hidden, train_private_labels)
    biased_classifier.fit(dev_hidden, dev_private_labels)
    # dev_leakage = biased_classifier.score(dev_hidden, dev_private_labels)
    dev_leakage = biased_classifier.score(dev_hidden, dev_private_labels)
    test_leakage = biased_classifier.score(test_hidden, test_private_labels)
    # print("Dev Accuracy: {}".format(dev_leakage))
    # print("Test Accuracy: {}".format(test_leakage))
    return test_leakage
