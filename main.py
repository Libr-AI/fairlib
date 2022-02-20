import logging
from src.base_options import options
from src import networks

if __name__ == '__main__':
    state = options.get_state()

    # Init the model
    model = networks.get_main_model(state)
    # state.opt.main_model = model
    logging.info('Model Initialized!')

    model.train_self()
    logging.info('Finished!')