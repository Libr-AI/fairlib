import logging

import sys
sys.path.append('../')

from src.base_options import options

if __name__ == '__main__':
    state = options.get_state()

    logging.info('train dataset size:\t{}'.format(len(state.opt.train_generator.dataset)))
    logging.info('validation dataset size: \t{}'.format(len(state.opt.dev_generator.dataset)))
    logging.info('test dataset size: \t{}'.format(len(state.opt.test_generator.dataset)))
    logging.info('datasets loaded!')