import sys
import torch
import logging
from .loaders import *
from . import utils
from collections import defaultdict

if sys.platform == "win32":
    default_dataset_roots = dict(
        Moji='D:\\Project\\User_gender_removal\\data\\deepmoji\\split2\\',
        Bios='D:\Project\Minding_Imbalance_in_Discriminator_Training\data\bios'
    )
else:
    default_dataset_roots = dict(
        Moji='/data/cephfs/punim1421/Dataset/deepmoji/split2/',
        Bios='./data/cephfs/punim1421/Dataset/bios_gender_economy'
    )

def get_dataloaders(args):
    assert args.dataset in ["Moji", "Bios"], "Not implemented"
    if args.dataset == "Moji":
        task_dataloader = DeepMojiDataset
    elif args.dataset == "Bios":
        task_dataloader = BiosDataset
    else:
        pass
    
    train_data = task_dataloader(args=args, split="train")
    dev_data = task_dataloader(args=args, split="dev")
    test_data = task_dataloader(args=args, split="test")


    # DataLoader Parameters
    tran_dataloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers}

    eval_dataloader_params = {
            'batch_size': args.test_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers}

    # init dataloader
    training_generator = torch.utils.data.DataLoader(train_data, **tran_dataloader_params)
    validation_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
    test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

    return training_generator, validation_generator, test_generator