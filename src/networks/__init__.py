import sys
import torch
import logging
from .classifier import MLP, BERTClassifier
from . import utils
from collections import defaultdict

def get_main_model(args):
    assert args.encoder_architecture in ["Fixed", "BERT", "DeepMoji"], "Not implemented"

    if args.encoder_architecture == "Fixed":
        model = MLP(args)
    elif args.encoder_architecture == "BERT":
        model = BERTClassifier(args)
    else:
        raise "not implemented yet"
    
    return model