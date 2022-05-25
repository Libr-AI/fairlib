import sys
import torch
import logging
from .loaders import *
from . import utils
from .encoder import text2id
from collections import defaultdict

if sys.platform == "win32":
    default_dataset_roots = dict(
        Moji= r'.\data\deepmoji',
        Bios_gender= r'.\data\bios',
        Bios_economy= r'.\data\bios',
        Bios_both= r'.\data\bios',
    )
else:
    default_dataset_roots = dict(
        Moji='./data/deepmoji',
        Bios_gender='./data/bios_gender_economy',
        Bios_economy='./data/bios_gender_economy',
        Bios_both='./data/bios_gender_economy',
    )

def get_dataloaders(args):
    """Initialize the torch dataloaders according to arguments.

    Args:
        args (namespace): arguments

    Raises:
        NotImplementedError: if correspoding components have not been implemented.

    Returns:
        tuple: dataloaders for training set, development set, and test set.
    """
    assert args.dataset in [
        "Sample", "test", "Moji", 
        "Bios_gender", "Bios_economy", "Bios_both",
        "Valence",
        ], "Not implemented"

    if args.dataset == "Moji":
        task_dataloader = DeepMojiDataset
    elif args.dataset in ["Bios_gender", "Bios_economy", "Bios_both"]:
        task_dataloader = BiosDataset
        args.protected_task = args.dataset.split("_")[1]
    elif args.dataset == "test":
        task_dataloader = TestDataset
    elif args.dataset == "Sample":
        task_dataloader = SampleDataset
    elif args.dataset == "Valence":
        task_dataloader = ValenceDataset
    else:
        pass
    
    if args.encoder_architecture == "Fixed":
        pass
    elif args.encoder_architecture == "BERT":
        # Init the encoder form text to idx.
        args.text_encoder = text2id(args)
    else:
        raise NotImplementedError

    train_data = task_dataloader(args=args, split="train")
    dev_data = task_dataloader(args=args, split="dev")
    test_data = task_dataloader(args=args, split="test")


    # DataLoader Parameters
    train_dataloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers}

    eval_dataloader_params = {
            'batch_size': args.test_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers}

    # init dataloader
    training_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
    validation_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
    test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

    return training_generator, validation_generator, test_generator