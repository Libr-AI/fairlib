import sys
import torch
import logging
from .classifier import MLP
from . import utils
from collections import defaultdict

def get_main_model(args):
    assert args.encoder_architecture in ["Fixed", "BERT", "DeepMoji"], "Not implemented"

    if args.encoder_architecture == "Fixed":
        model = MLP(args)
    else:
        raise "not implemented yet"
    
    return model