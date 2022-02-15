import sys
import torch
import logging
from .loaders import *
from . import utils
from collections import defaultdict

def get_dataloaders(args):
    assert args.task in ["Moji", "Bios"], "Not implemented"
    if args.task == "Moji":
        task_dataloader = DeepMojiDataset
    elif args.task == "Bios":
        task_dataloader = BiosDataset
    else:
        pass
    
    train_dataloader = task_dataloader(args=args, split="train")
    dev_dataloader = task_dataloader(args=args, split="dev")
    test_dataloader = task_dataloader(args=args, split="test")

    return train_dataloader, dev_dataloader, test_dataloader