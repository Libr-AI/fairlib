import logging

import sys
sys.path.append('../')

from src.base_options import options
from src import networks

if __name__ == '__main__':
    # state = options.get_state(conf_file=r"D:\Project\Fair_NLP_Classification\results\vanilla\Bios_gender\0\opt.yaml")
    state = options.get_state()

    logging.info('train dataset size:\t{}'.format(len(state.opt.train_generator.dataset)))
    logging.info('validation dataset size: \t{}'.format(len(state.opt.dev_generator.dataset)))
    logging.info('test dataset size: \t{}'.format(len(state.opt.test_generator.dataset)))
    logging.info('datasets loaded!')

    # Init the model
    model = networks.get_main_model(state)
    # state.opt.main_model = model
    logging.info('Model Initialized!')

    model.train_self()
    logging.info('Finished!')