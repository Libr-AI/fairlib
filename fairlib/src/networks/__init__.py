import sys
import torch
import logging
from .classifier import MLP, BERTClassifier,ConvNet
from . import utils
from . import INLP
from . import FairCL
from . import DyBT
from . import adv
from collections import defaultdict

def get_main_model(args):

    if args.encoder_architecture == "Fixed":
        model = MLP(args)
    elif args.encoder_architecture == "BERT":
        model = BERTClassifier(args)
    elif args.encoder_architecture == "MNIST":
        model = ConvNet(args)
    else:
        raise NotImplementedError
    
    return model