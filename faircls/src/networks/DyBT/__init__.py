from .fairbatch_sampler import FairBatch
from .generalized_fairbatch import Generalized_FairBatch
from .gdl import Group_Difference_Loss

def init_sampler(model, args):
    assert args.DyBTObj in ["joint", "y", "g", "stratified_y", "stratified_g"]

    if args.DyBT == 'FairBatch':
        # Init the fairbatch sampler
        DyBT_sampler = FairBatch(model, args)
    elif args.DyBT == 'GeneralizedFB':
        DyBT_sampler = Generalized_FairBatch(model, args)
    return DyBT_sampler