import argparse
import os
import torch
import yaml
import time
import numpy as np
import random
import logging
from contextlib import contextmanager

from src import utils
from src import dataloaders
from src import networks
from src.networks import adv

class State(object):

    class UniqueNamespace(argparse.Namespace):
        def __init__(self, requires_unique=True):
            self.__requires_unique = requires_unique
            self.__set_value = {}

        def requires_unique(self):
            return self.__requires_unique

        def mark_set(self, name, value):
            if self.__requires_unique and name in self.__set_value:
                raise argparse.ArgumentTypeError(
                    "'{}' appears several times: {}, {}.".format(
                        name, self.__set_value[name], value))
            self.__set_value[name] = value


    __inited = False

    def __init__(self, opt=None):
        if opt is None:
            self.opt = self.UniqueNamespace()
        else:
            if isinstance(opt, argparse.Namespace):
                opt = vars(opt)
            self.opt = argparse.Namespace(**opt)
        self.extras = {}
        self.__inited = True
        self._output_flag = True

    def __setattr__(self, k, v):
        if not self.__inited:
            return super(State, self).__setattr__(k, v)
        else:
            self.extras[k] = v

    def __getattr__(self, k):
        if k in self.extras:
            return self.extras[k]
        elif k in self.opt:
            return getattr(self.opt, k)
        raise AttributeError(k)

    def copy(self):
        return argparse.Namespace(**self.merge())

    def get_output_flag(self):
        return self._output_flag

    @contextmanager
    def pretend(self, **kwargs):
        saved = {}
        for key, val in kwargs.items():
            if key in self.extras:
                saved[key] = self.extras[key]
            setattr(self, key, val)
        yield
        for key, val in kwargs.items():
            self.pop(key)
            if key in saved:
                self.extras[key] = saved[key]

    def set_output_flag(self, val):
        self._output_flag = val

    def pop(self, k, default=None):
        return self.extras.pop(k, default)

    def clear(self):
        self.extras.clear()

    # returns a single dict containing both opt and extras
    def merge(self, public_only=False):
        vs = vars(self.opt).copy()
        vs.update(self.extras)
        if public_only:
            for k in tuple(vs.keys()):
                if k.startswith('_'):
                    vs.pop(k)
        return vs

    def get_base_directory(self):
        vs = self.merge()
        opt = argparse.Namespace(**vs)
        dirs = [opt.project_dir, opt.dataset, opt.exp_id]
        return os.path.join(opt.results_dir, *dirs)

    def get_model_directory(self):
        vs = self.merge()
        opt = argparse.Namespace(**vs)
        dirs = [opt.project_dir, opt.dataset, opt.exp_id, opt.model_dir]
        return os.path.join(opt.results_dir, *dirs)

class BaseOptions(object):
    def __init__(self):
        # argparse utils

        def comp(type, op, ref):
            op = getattr(type, '__{}__'.format(op))

            def check(value):
                ivalue = type(value)
                if not op(ivalue, ref):
                    raise argparse.ArgumentTypeError("expected value {} {}, but got {}".format(op, ref, value))
                return ivalue

            return check

        def int_gt(i):
            return comp(int, 'gt', i)

        def float_gt(i):
            return comp(float, 'gt', i)

        pos_int = int_gt(0)
        nonneg_int = int_gt(-1)
        pos_float = float_gt(0)

        def get_unique_action_cls(actual_action_cls):
            class UniqueSetAttrAction(argparse.Action):
                def __init__(self, *args, **kwargs):
                    self.subaction = actual_action_cls(*args, **kwargs)

                def __call__(self, parser, namespace, values, option_string=None):
                    if isinstance(namespace, State.UniqueNamespace):
                        requires_unique = namespace.requires_unique()
                    else:
                        requires_unique = False
                    if requires_unique:
                        namespace.mark_set(self.subaction.dest, values)
                    self.subaction(parser, namespace, values, option_string)

                def __getattr__(self, name):
                    return getattr(self.subaction, name)

            return UniqueSetAttrAction

        self.parser = parser = argparse.ArgumentParser(description='Fair NLP Classification')

        action_registry = parser._registries['action']
        for name, action_cls in action_registry.items():
            action_registry[name] = get_unique_action_cls(action_cls)

        parser.add_argument('--project_dir', type=str, default=None,
                            help='dataset root')
        parser.add_argument('--exp_id', type=str, default="test",
                            help='experiment id')
        parser.add_argument('--encoder_architecture', type=str, default="Fixed",
                            help='Fixed | BERT | DeepMoji')
        parser.add_argument('--batch_size', type=pos_int, default=1024,
                            help='input batch size for training (default: 1024)')
        parser.add_argument('--test_batch_size', type=pos_int, default=1024,
                            help='input batch size for testing (default: 1024)')
        parser.add_argument('--epochs', type=pos_int, default=100, metavar='N',
                            help='number of total epochs to train (default: 100)')
        parser.add_argument('--lr', type=pos_float, default=0.01, metavar='LR',
                            help='learning rate used to actually learn stuff (default: 0.01)')
        parser.add_argument('--epochs_since_improvement', type=pos_int, default=5,
                            help='terminate training for early stopping')
        parser.add_argument('--base_seed', type=int, default=1, metavar='S',
                            help='base random seed (default: 1)')
        parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--checkpoint_interval', type=int, default=1, metavar='N',
                            help='checkpoint interval (epoch)')
        parser.add_argument('--dataset', type=str, default='Moji',
                            help='dataset: Moji | Bios_gender | Bios_economy | Bios_both')
        parser.add_argument('--data_dir', type=str, default=None,
                            help='dataset root')
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='results directory')
        parser.add_argument('--model_dir', type=str, default='./models/',
                            help='directory storing trained models')
        parser.add_argument('--device_id', type=comp(int, 'ge', -1), default=0, help='device id, -1 is cpu')
        parser.add_argument('--num_workers', type=nonneg_int, default=0,
                            help='number of data loader workers')
        parser.add_argument('--no_log', action='store_true',
                            help='if set, will not log into file')
        parser.add_argument('--log_level', type=str, default='INFO',
                            help='logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL')

        # Arguments for the main task model
        parser.add_argument('--hidden_size',  type=pos_int, default=300, 
                            help='number of hidden units for the main task classifier')
        parser.add_argument('--n_hidden',  type=int, default=2, 
                            help='number of hidden layers')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout probability')
        parser.add_argument('--emb_size', type=pos_int, default=2304,
                            help='input embedding dimension')
        parser.add_argument('--num_classes', type=pos_int, default=2,
                            help='number of target classes')
        parser.add_argument('--activation_function', type=str, default="ReLu",
                            help='nonlinear activation function for the main task model')
        parser.add_argument('--batch_norm',  action='store_true', default=False, 
                            help='apply 1d batch norm to the model')

        # Arguments for fair training
        parser.add_argument('--BT', type=str, default=None, help='Reweighting | Resampling')
        parser.add_argument('--BTObj', type=str, default=None, help='joint | y | g | stratified_y | stratified_g')
        parser.add_argument('--full_label',  action='store_true', default=True, help='require full protected label')
        parser.add_argument('--gated',  action='store_true', default=False, 
                            help='gated model for augmented inputs given protected labels')
        parser.add_argument('--gated_mapping',  type=str, default=None)

        

        # Arguments for adversarial debiasing
        parser.add_argument('--adv_debiasing', type=bool, default=False, help='Adv debiasing?')
        # The following arguments will only be used if adv_debiasing is set to True
        parser.add_argument('--adv_update_frequency', type=str, default="Epoch", help='Epoch | Batch')
        parser.add_argument('--adv_checkpoint_interval', type=int, default=1, metavar='N',
                            help='checkpoint interval (epoch)')
        parser.add_argument('--adv_level', type=str, default="last_hidden", help='"input | last_hidden | output')
        parser.add_argument('--adv_batch_size', type=pos_int, default=1024,
                            help='input batch size for discriminator training (default: 1024)')
        parser.add_argument('--adv_test_batch_size', type=pos_int, default=1024,
                            help='input batch size for discriminator testing (default: 1024)')
        parser.add_argument('--adv_epochs', type=pos_int, default=100, metavar='N',
                            help='number of total epochs to train the discriminator (default: 100)')
        parser.add_argument('--adv_lr', type=pos_float, default=0.01, metavar='LR',
                            help='learning rate used to actually learn adversarial stuff (default: 0.01)')
        parser.add_argument('--adv_epochs_since_improvement', type=pos_int, default=5,
                            help='terminate discriminator training for early stopping')
        parser.add_argument('--adv_lambda', type=float, default=0.8, help='strength of adversarial regularization')
        parser.add_argument('--adv_hidden_size',  type=pos_int, default=300, 
                            help='number of hidden units for the adversarial discriminator')
        parser.add_argument('--adv_n_hidden',  type=int, default=2, 
                            help='number of hidden layers of the discriminator')
        parser.add_argument('--adv_dropout', type=float, default=0,
                            help='dropout probability in the discriminator')
        parser.add_argument('--adv_num_classes', type=pos_int, default=2,
                            help='number of protected classes')
        parser.add_argument('--adv_activation_function', type=str, default="ReLu",
                            help='nonlinear activation function for the discriminator')
        parser.add_argument('--adv_batch_norm',  action='store_true', default=False, 
                            help='apply 1d batch norm to the discriminator')

        # Gated adv
        parser.add_argument('--adv_gated',  action='store_true', default=False, 
                            help='gated discriminator for augmented inputs given target labels')
        parser.add_argument('--adv_gated_mapping',  type=str, default=None)
        # Diverse
        parser.add_argument('--adv_num_subDiscriminator', type=pos_int, default=1,
                            help='number of subdiscriminators. 1 is the standard setting.')
        parser.add_argument('--adv_diverse_lambda', type=float, default=0.0, 
                            help='strength of difference loss to encourage diverse representations for ensemble adv.')
        
        # Decoupling adversarial training
        parser.add_argument('--adv_decoupling', action='store_true', default=False,
                            help='decoupling the training and regularization of the adv discriminator and the main task model.')

        # Use uniform soft-labels for the adversarial regularization
        parser.add_argument('--adv_uniform_label', action='store_true', default=False,
                            help='Using uniform soft-labels for the adversarial regularization')



    def get_dummy_state(self, *cmdargs, yaml_file=None, **opt_pairs):
        if yaml_file is None:
            # Use default Namespace (not UniqueNamespace) because dummy state may
            # want to overwrite things using `cmdargs`
            opt = self.parser.parse_args(args=list(cmdargs), namespace=argparse.Namespace())
        else:
            with open(yaml_file, 'r') as f:
                opt = yaml.load(f)
        state = State(opt)
        valid_keys = set(state.merge().keys())
        for k in opt_pairs:
            # TODO: check against argparse instead
            assert k in valid_keys, "'{}' is not a valid key".format(k)
        state.extras.update(opt_pairs)
        return self.set_state(state, dummy=True)

    def get_state(self):
        if hasattr(self, 'state'):
            return self.state

        logging.getLogger().setLevel(logging.DEBUG)
        self.opt, unknowns = self.parser.parse_known_args(namespace=State.UniqueNamespace())
        assert len(unknowns) == 0, 'Unexpected args: {}'.format(unknowns)
        self.state = State(self.opt)
        return self.set_state(self.state)

    def set_state(self, state, dummy=False):

        base_dir = state.get_base_directory()

        state.opt.model_dir = state.get_model_directory()

        state.opt.start_time = time.strftime(r"%Y-%m-%d %H:%M:%S")

        state.set_output_flag(not dummy)
        
        if not dummy:
            utils.mkdir(base_dir)

            utils.mkdir(state.opt.model_dir)

            # First thing: set logging config:
            if not state.opt.no_log:
                log_filename = 'output'
                log_filename += '.log'
                state.opt.log_file = os.path.join(base_dir, log_filename)
            else:
                state.opt.log_file = None

            state.opt.log_level = state.opt.log_level.upper()

            logging_prefix = ''
            utils.logging.configure(
                state.opt.log_file, 
                getattr(logging, state.opt.log_level),
                prefix=logging_prefix
                )

            logging.info("=" * 40 + " " + state.opt.start_time + " " + "=" * 40)
            logging.info('Base directory is {}'.format(base_dir))
            
        # Write yaml
        yaml_str = yaml.dump(state.merge(public_only=True), default_flow_style=False, indent=4)
        logging.info("Options:\n\t" + yaml_str.replace("\n", "\n\t"))

        if state.get_output_flag():
            yaml_name = os.path.join(base_dir, 'opt.yaml')
            if os.path.isfile(yaml_name):
                old_opt_dir = os.path.join(base_dir, 'old_opts')
                utils.mkdir(old_opt_dir)
                with open(yaml_name, 'r') as f:
                    # ignore unknown ctors
                    yaml.add_multi_constructor('', lambda loader, suffix, node: None)
                    old_yaml = yaml.full_load(f)  # this is a dict
                old_yaml_time = old_yaml.get('start_time', 'unknown_time')
                for c in ':-':
                    old_yaml_time = old_yaml_time.replace(c, '_')
                old_yaml_time = old_yaml_time.replace(' ', '__')
                old_opt_new_name = os.path.join(old_opt_dir, 'opt_{}.yaml'.format(old_yaml_time))
                try:
                    os.rename(yaml_name, old_opt_new_name)
                    logging.warning('{} already exists, moved to {}'.format(yaml_name, old_opt_new_name))
                except FileNotFoundError:
                    logging.warning((
                        '{} already exists, tried to move to {}, but failed, '
                        'possibly due to other process having already done it'
                    ).format(yaml_name, old_opt_new_name))
                    pass

            with open(yaml_name, 'w') as f:
                f.write(yaml_str)

        if state.device_id < 0:
            state.opt.device = torch.device("cpu")
        else:
            torch.cuda.set_device(state.device_id)
            state.opt.device = torch.device("cuda:{}".format(state.device_id))

        if not dummy:
            if state.device.type == 'cuda' and torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = True

            seed = state.base_seed
            state.opt.seed = seed

            # torch.manual_seed will seed ALL GPUs.
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Init the dataloaders
            if state.data_dir is None:
                state.data_dir = dataloaders.default_dataset_roots[state.dataset]
            train_iterator, dev_iterator, test_iterator = dataloaders.get_dataloaders(state)

            state.opt.train_generator = train_iterator
            state.opt.dev_generator = dev_iterator
            state.opt.test_generator = test_iterator

            logging.info('train dataset size:\t{}'.format(len(train_iterator.dataset)))
            logging.info('validation dataset size: \t{}'.format(len(dev_iterator.dataset)))
            logging.info('test dataset size: \t{}'.format(len(test_iterator.dataset)))
            logging.info('datasets built!')

            if state.adv_debiasing and state.adv_decoupling:
                raise NotImplementedError

        return state


options = BaseOptions()
