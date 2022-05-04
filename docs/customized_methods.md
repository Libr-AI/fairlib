# Customized methods

Debiasing methods in *fairlib* should be connected the `BaseModel` class, such that these these methods generalize to any model architectures. There are different ways of introducing new methods to *fairlib*, and here we just show on of them. 

## Step 1: Hyperparameters

Taking the balanced training method (BT) as an example, we need to specify objective for BT and the way of achieving desired objectives.

These options (hyperparameters) are introduced in the `fairlib/src/base_options.py`.

```python
        parser.add_argument('--BT', type=str, default=None, help='Reweighting | Resampling')
        parser.add_argument('--BTObj', type=str, default=None, help='joint | y | g | stratified_y | stratified_g')
```

For consistency, our suggestions for defining method-specific hyperparameters includes:
- By default, the debiasing method is not employed during training. We could either set the default value as `None` as shown for BT, or store the decision as a bool type, for example,
  ```python
    parser.add_argument('--adv_debiasing', action='store_true', default=False, help='Adv debiasing?')
  ```
- For better readability, use string type options.
- Use the same prefix for the same set of hyperparameters.

## Step 2: Methods Implementations

### Pre-processing Methods
Pre-processing methods manipulate data distributions at the beginning, so these methods could either be applied outside of the process *fairlib* statically when preparing dataset, or be combined with the dataloader initialization. 

To be incorporated with dataloader initialization, such methods should be integrated with the `BaseDataset` class as defined in `fairlib/src/dataloaders/utils.py`.

### At-training Methods
This type of method generally introduces objectives or regularizations in addition to the main model training. In *fairlib*, we have implemented such methods as a child class of `torch.nn.Module`, which could be integrated with automatic differentiation.

For a simplest example, please see the implementation of fair supervised contrastive learning method (FairSCL) as an example. 

`fairlib/src/networks/FairCL/`

Please create a new dir within the `networks` dir, e.g., `fairlib/src/networks/NEW_METHODS`, and include all related files in that dir.

To be integrated with *fairlib*, corresponding methods should be added to the `train_epoch` function (`fairlib/src/networks/utils.py`), which trains the the main model one epoch. 

```python
        if args.FCL:
            # get hidden representations
            if args.gated:
                hs = model.hidden(text, p_tags)
            else:
                hs = model.hidden(text)


            # update the loss with Fair Supervised Contrastive Loss
            fscl_loss = args.FairSCL(hs, tags, p_tags)
            loss = loss + fscl_loss
```

### Post-processing Methods
Post-processing method implementations should also be included in `fairlib/src/networks/NEW_METHODS`.

Taking the INLP method as an example, related codes can be found from `fairlib/src/networks/INLP`

Different ot pre-processing and at-training, post-processing methods need to perform additional evaluations. which could be done by the built-in function
```python
from fairlib.evaluators import present_evaluation_scores
        present_evaluation_scores(
            valid_preds = dev_y_pred, valid_labels = dev_labels, 
            valid_private_labels = dev_private_labels,
            test_preds = test_y_pred, test_labels = test_labels, 
            test_private_labels = test_private_labels,
            epoch = iteration, epochs_since_improvement = None, 
            model = model, epoch_valid_loss = None,
            is_best = False, prefix = "INLP_checkpoint",
            )
```

This function will evaluate valid and test scores, save checkpoints, and print scores corresponding to a epoch.

To be integrated with *fairlib*, a post-processing method should be employed after the model training. Thus, we will employ this method apart from the model training.

In the `fairlib/__main__.py` file, we first init the options and train a model. After which, we prot-process the trained model as follows:
```python
def main():
    options = BaseOptions()
    state = options.get_state()


    # Init the model
    model = networks.get_main_model(state)
    # state.opt.main_model = model
    logging.info('Model Initialized!')


    model.train_self()
    logging.info('Model Trained!')


    if state.INLP:
        logging.info('Run INLP')
        from src.networks.INLP import get_INLP_trade_offs
        get_INLP_trade_offs(model, state)


    logging.info('Finished!')
```