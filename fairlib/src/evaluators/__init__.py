from .evaluator import *
from .utils import *

def present_evaluation_scores(
    valid_preds, valid_labels, valid_private_labels,
    test_preds, test_labels, test_private_labels,
    epoch, epochs_since_improvement, model, epoch_valid_loss,
    is_best, 
    ):
    valid_scores, valid_confusion_matrices = gap_eval_scores(
        y_pred=valid_preds,
        y_true=valid_labels, 
        protected_attribute=valid_private_labels)
                
    test_scores, test_confusion_matrices = gap_eval_scores(
        y_pred=test_preds,
        y_true=test_labels, 
        protected_attribute=test_private_labels)

    # Save checkpoint
    save_checkpoint(
        epoch = epoch, 
        epochs_since_improvement = epochs_since_improvement, 
        model = model, 
        loss = epoch_valid_loss, 
        dev_predictions = valid_preds, 
        test_predictions = test_preds,
        dev_evaluations = valid_scores, 
        test_evaluations = test_scores,
        valid_confusion_matrices = valid_confusion_matrices,
        test_confusion_matrices = test_confusion_matrices,
        is_best = is_best,
        checkpoint_dir = model.args.model_dir)

    validation_results = ["{}: {:2.2f}\t".format(k, 100.*valid_scores[k]) for k in valid_scores.keys()]
    logging.info(('Validation {}').format("".join(validation_results)))
    Test_results = ["{}: {:2.2f}\t".format(k, 100.*test_scores[k]) for k in test_scores.keys()]
    logging.info(('Test {}').format("".join(Test_results)))