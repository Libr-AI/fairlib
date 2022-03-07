import logging

import sys
sys.path.append('../')

from src.base_options import options
from src import networks
from src.networks.fairbatch import FairBatch

if __name__ == '__main__':
    state = options.get_state()

    logging.info('train dataset size:\t{}'.format(len(state.opt.train_generator.dataset)))
    logging.info('validation dataset size: \t{}'.format(len(state.opt.dev_generator.dataset)))
    logging.info('test dataset size: \t{}'.format(len(state.opt.test_generator.dataset)))
    logging.info('datasets loaded!')

    # Init the model
    model = networks.get_main_model(state)
    # state.opt.main_model = model
    logging.info('Model Initialized!')

    if state.DyBT is not None and state.DyBT == 'FairBatch':
        # Init the fairbatch sampler
        fairbatch_sampler = FairBatch(model, state)
        # Replace the tran iterator with fairbatch version
        import torch
        state.opt.train_generator = torch.utils.data.DataLoader(state.opt.train_generator.dataset, sampler=fairbatch_sampler, num_workers=0)


    model.train_self()
    logging.info('Finished!')