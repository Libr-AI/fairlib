import argparse
import os
import torch
import yaml
import time
import numpy as np
import random
import logging
from contextlib import contextmanager
import sys
import traceback

from . import utils
from . import dataloaders
from . import networks
from .networks import adv
from .networks import FairCL
from .networks.DyBT import Group_Difference_Loss
from .networks.ARL import ARL
from .networks.knn_labels import KNN_Loss

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

        parser.add_argument('--project_dir', type=str, default="dev",
                            help='protect step root: dev | hypertune | final')
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
        parser.add_argument('--lr', type=pos_float, default=0.003, metavar='LR',
                            help='learning rate used to actually learn stuff (default: 0.003)')
        parser.add_argument('--weight_decay', type=float, default=0.0,
                            help='weight decay (L2 penalty) (default: 0)')
        parser.add_argument('--epochs_since_improvement', type=pos_int, default=5,
                            help='terminate training for early stopping')
        parser.add_argument('--base_seed', type=int, default=1, metavar='S',
                            help='base random seed (default: 1)')
        parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save_batch_results', action='store_true', default=False, 
                            help='if saving batch evaluation results')
        parser.add_argument('--save_models', action='store_true', default=False, 
                            help='if saving model parameters')
        parser.add_argument('--checkpoint_interval', type=int, default=1, metavar='N',
                            help='checkpoint interval (epoch)')
        parser.add_argument('--dataset', type=str, default='Moji',
                            help='dataset: Moji | Bios_gender | Bios_economy | Bios_both')
        parser.add_argument('--data_dir', type=str, default=None,
                            help='dataset root')
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='results directory')
        parser.add_argument('--model_dir', type=str, default='models',
                            help='directory storing trained models')
        parser.add_argument('--device_id', type=comp(int, 'ge', -1), default=0, help='device id, -1 is cpu')
        parser.add_argument('--num_workers', type=nonneg_int, default=0,
                            help='number of data loader workers')
        parser.add_argument('--no_log', action='store_true',
                            help='if set, will not log into file')
        parser.add_argument('--log_level', type=str, default='INFO',
                            help='logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL')
        parser.add_argument('--conf_file', type=str, default=None,
                            help='path to the YAML file for reproduce an an experiment')

        # Regression related arguments
        parser.add_argument('--regression',  action='store_true', default=False, 
                            help='indicate the downstream task is regression')
        parser.add_argument('--n_bins',  type=int, default=4, 
                            help='number of bins for discretizing proxy labels')

        # Handle iPython arguments
        parser.add_argument('--f', type=str, default=None, help='path to the YAML file for reproduce an an experiment')

        # Arguments for the main task model
        parser.add_argument('--hidden_size',  type=pos_int, default=300, 
                            help='number of hidden units for the main task classifier')
        parser.add_argument('--n_hidden',  type=int, default=2, 
                            help='number of hidden layers')
        parser.add_argument('--dropout', type=float, default=0,
                            help='dropout probability')
        parser.add_argument('--emb_size', type=pos_int, default=2304,
                            help='input embedding dimension')
        parser.add_argument('--num_classes', type=pos_int, default=2,
                            help='number of target classes')
        parser.add_argument('--num_groups', type=pos_int, default=2,
                            help='number of protected classes')
        parser.add_argument('--activation_function', type=str, default="Tanh",
                            help='nonlinear activation function for the main task model')
        parser.add_argument('--batch_norm',  action='store_true', default=False, 
                            help='apply 1d batch norm to the model')
        parser.add_argument('--classification_head_update_frequency',  type=pos_int, default=1,
                            help='the update frequency of the main model classification head (every N batches)')

        # Arguments for balanced training
        parser.add_argument('--BT', type=str, default=None, help='Reweighting | Resampling | Downsampling')
        parser.add_argument('--BTObj', type=str, default=None, help='joint | y | g | stratified_y | stratified_g | EO')
        parser.add_argument('--full_label',  action='store_true', default=True, help='require full protected label')

        # Arguments for augmented training
        parser.add_argument('--gated',  action='store_true', default=False, 
                            help='gated model for augmented inputs given protected labels')
        parser.add_argument('--gated_mapping',  type=str, default=None)

        # Arguments for dynamic balanced training
        parser.add_argument('--DyBT', type=str, default=None, help='FairBatch | GroupDifference | GeneralizedFB')
        parser.add_argument('--DyBTObj', type=str, default=None, help='joint | y | g | stratified_y | stratified_g | EO')
        parser.add_argument('--DyBTalpha', type=float, default=0.1, help='a positive number for dynamic adjustment.')
        parser.add_argument('--DyBTinit', type=str, default="original", help='original | balanced')

        # Arguments for adversarial debiasing
        parser.add_argument('--adv_debiasing', action='store_true', default=False, help='Adv debiasing?')
        # The following arguments will only be used if adv_debiasing is set to True
        parser.add_argument('--adv_update_frequency', type=str, default="Batch", help='Epoch | Batch')
        parser.add_argument('--adv_checkpoint_interval', type=int, default=1, metavar='N',
                            help='checkpoint interval (epoch)')
        parser.add_argument('--adv_level', type=str, default="last_hidden", help='input | last_hidden | output')
        parser.add_argument('--adv_batch_size', type=pos_int, default=1024,
                            help='input batch size for discriminator training (default: 1024)')
        parser.add_argument('--adv_test_batch_size', type=pos_int, default=1024,
                            help='input batch size for discriminator testing (default: 1024)')
        parser.add_argument('--adv_epochs', type=pos_int, default=100, metavar='N',
                            help='number of total epochs to train the discriminator (default: 100)')
        parser.add_argument('--adv_lr', type=pos_float, default=0.001, metavar='LR',
                            help='learning rate used to actually learn adversarial stuff (default: 0.01)')
        parser.add_argument('--adv_epochs_since_improvement', type=pos_int, default=5,
                            help='terminate discriminator training for early stopping')
        parser.add_argument('--adv_lambda', type=float, default=1, help='strength of adversarial regularization')
        parser.add_argument('--adv_hidden_size',  type=pos_int, default=300, 
                            help='number of hidden units for the adversarial discriminator')
        parser.add_argument('--adv_n_hidden',  type=int, default=2, 
                            help='number of hidden layers of the discriminator')
        parser.add_argument('--adv_dropout', type=float, default=0,
                            help='dropout probability in the discriminator')
        parser.add_argument('--adv_activation_function', type=str, default="ReLu",
                            help='nonlinear activation function for the discriminator')
        parser.add_argument('--adv_batch_norm',  action='store_true', default=False, 
                            help='apply 1d batch norm to the discriminator')

        # Gated adv
        parser.add_argument('--adv_gated',  action='store_true', default=False, 
                            help='gated discriminator for augmented inputs given target labels')
        parser.add_argument('--adv_gated_type', type=str, default="Augmentation",
                            help='Augmentation | Inputs | Separate')
        parser.add_argument('--adv_BT', type=str, default=None, help='instacne reweighting for adv')
        parser.add_argument('--adv_BTObj', type=str, default=None, help='instacne reweighting for adv')

        parser.add_argument('--adv_gated_mapping',  type=str, default=None)
        # Diverse
        parser.add_argument('--adv_num_subDiscriminator', type=pos_int, default=1,
                            help='number of subdiscriminators. 1 is the standard setting.')
        parser.add_argument('--adv_diverse_lambda', type=float, default=0.0, 
                            help='strength of difference loss to encourage diverse representations for ensemble adv.')
        
        # Decoupling adversarial training
        parser.add_argument('--adv_decoupling', action='store_true', default=False,
                            help='decoupling the training and regularization of the adv discriminator and the main task model.')
        # Decoupling adversarial training
        parser.add_argument('--adv_decoupling_labelled_proportion', type=pos_float, default=1.0,
                            help='proportion of instances that are labelled with protected labels. If the original protected labels have already included NAN, set this to 1.')

        # Use uniform soft-labels for the adversarial regularization
        parser.add_argument('--adv_uniform_label', action='store_true', default=False,
                            help='Using uniform soft-labels for the adversarial regularization')

        # INLP
        parser.add_argument('--INLP', action='store_true', default=False,
                            help='Perform INLP')
        parser.add_argument("--INLP_discriminator_reweighting", type=str, default=None, 
                            help='if train the linear discriminator with reweighting')
        parser.add_argument("--INLP_by_class", action='store_true', default=False,
                            help="estimate the nullspace by_class")
        parser.add_argument("--INLP_n", type=pos_int, default=300,
                            help="the maximum number of null-space projection iteration")
        parser.add_argument("--INLP_min_acc", type=float, default=0.0,
                            help="ignore the iteration if the acc is lower than the threshold")

        # Fair Supervised Contrastive Learning
        parser.add_argument("--FCL",action='store_true', default=False,
                            help='Perform Fair Supervised Contrastive Learning')
        parser.add_argument('--FCLObj', type=str, default="g", help='g | EO')
        parser.add_argument("--fcl_temperature_y", type=float, default=0.01,
                            help="temperature for the fcl wrt main task learning")
        parser.add_argument("--fcl_temperature_g", type=float, default=0.01,
                            help="temperature for the fcl wrt protected attribute unlearning")
        parser.add_argument("--fcl_base_temperature_y", type=float, default=0.01,
                            help="base temperature for the fcl wrt main task learning")
        parser.add_argument("--fcl_base_temperature_g", type=float, default=0.01,
                            help="base temperature for the fcl wrt protected attribute unlearning")
        parser.add_argument("--fcl_lambda_y", type=float, default=0.1,
                            help="strength of the supervised contrastive loss")
        parser.add_argument("--fcl_lambda_g", type=float, default=0.1,
                            help="strength of the fair supervised contrastive loss")

        # Manipulate data distribution directly after data loading by interpolating distributions.
        parser.add_argument('--GBT', action="store_true", default=False, help='whether or not manipulate loaded data distribution')
        parser.add_argument('--GBTObj', type=str, default=None, help='joint | y | g | y_cond_g | g_cond_y')
        parser.add_argument('--GBT_N', type=nonneg_int, default=None, help='size of the manipulated dataset')
        parser.add_argument("--GBT_alpha", type=float, default=1, help="interpolation for generalized BT")


        # KNN labels
        parser.add_argument("--knn_labels",action='store_true', default=False,
                            help='Replace actual protected labels with knn_labels')
        parser.add_argument('--knn_labels_k', type=nonneg_int, default=5, help='number of instances for knn')
        parser.add_argument('--knn_labels_p', type=nonneg_int, default=2, help='power of the distance metric for knn')

        # KNN label debiasing
        parser.add_argument('--UKNN_debiasing', action='store_true', default=False,
                            help='Perform simple debiasing based on KNN proxy labels')

        parser.add_argument("--UKNN_lambda", type=float, default=0, 
                            help="weights of the UKNN regularization")

        # ARL
        parser.add_argument('--ARL', action='store_true', default=False,
                            help='Perform adversarial reweighted learning (ARL)')
        parser.add_argument('--ARL_n',type=pos_int, default=1,
                            help='Update the adversary n times per main model update')


    def get_dummy_state(self, *cmdargs, yaml_file=None, **opt_pairs):
        if yaml_file is None:
            # Use default Namespace (not UniqueNamespace) because dummy state may
            # want to overwrite things using `cmdargs`
            opt = self.parser.parse_args(args=list(cmdargs), namespace=argparse.Namespace())
        else:
            with open(yaml_file, 'r') as f:
                opt = yaml.full_load(f)
        state = State(opt)
        valid_keys = set(state.merge().keys())
        for k in opt_pairs:
            # TODO: check against argparse instead
            assert k in valid_keys, "'{}' is not a valid key".format(k)
        state.extras.update(opt_pairs)
        return self.set_state(state, dummy=True)

    def get_state(self, args:dict={}, conf_file=None, silence=False):
        """get state from yaml and args

        Args:
            args (dict, optional): arguments. Defaults to {}.
            conf_file (_type_, optional): yaml file path. Defaults to None.

        Returns:
            agrs.NameSapce: aruguments for training
        """

        if hasattr(self, 'state'):
            return self.state

        logging.getLogger().setLevel(logging.DEBUG)
        opt, unknowns = self.parser.parse_known_args(namespace=State.UniqueNamespace())
        # assert len(unknowns) == 0, 'Unexpected args: {}'.format(unknowns)
        if len(unknowns) != 0:
            logging.info('Unexpected args: {}'.format(unknowns))
        opt = vars(opt)
        
        if conf_file is not None:
            with open(conf_file, 'r') as f:
                yaml_opt = yaml.full_load(f)
            opt.update(yaml_opt)
        else:
            if opt["conf_file"] is not None:
                with open(opt["conf_file"], 'r') as f:
                    yaml_opt = yaml.full_load(f)
                opt.update(yaml_opt)
        
        opt.update(args)
        
        self.opt = argparse.Namespace(**opt)
        self.state = State(self.opt)
        return self.set_state(self.state, silence=silence)

    def set_state(self, state, dummy=False, silence=False):

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
        if not silence:
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

            # Init for regression
            if state.regression:
                # Set the number of classes based on bin labels.
                state.num_classes = state.n_bins

            # Init the dataloaders
            if state.data_dir is None:
                state.data_dir = dataloaders.default_dataset_roots[state.dataset]
            try:
                train_iterator, dev_iterator, test_iterator = dataloaders.get_dataloaders(state)

                state.opt.train_generator = train_iterator
                state.opt.dev_generator = dev_iterator
                state.opt.test_generator = test_iterator

                if not silence:
                    logging.info('train dataset size:\t{}'.format(len(train_iterator.dataset)))
                    logging.info('validation dataset size: \t{}'.format(len(dev_iterator.dataset)))
                    logging.info('test dataset size: \t{}'.format(len(test_iterator.dataset)))
                    logging.info('datasets built!')
            except Exception as e:
                # Get current system exception
                ex_type, ex_value, ex_traceback = sys.exc_info()
                # Extract unformatter stack traces as tuples
                trace_back = traceback.extract_tb(ex_traceback)
                # Format stacktrace
                stack_trace = list()
                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                
                logging.info("Exception type : %s " % ex_type.__name__)
                logging.info("Exception message : %s" %ex_value)
                logging.info("Stack trace : %s" %stack_trace)
                
                logging.info('dataloaders need to be initialized!')
            
            # Init discriminator for adversarial training
            if state.adv_debiasing:
                # if state.adv_decoupling:
                #     raise NotImplementedError

                if state.adv_gated and (state.adv_gated_type == "Separate"):
                    # Train a set of discriminators for each class
                    state.opt.discriminator = [networks.adv.Discriminator(state) for _ in range(state.num_classes)]
                else:
                    # All other adv settings
                    state.opt.discriminator = networks.adv.Discriminator(state)
                logging.info('Discriminator built!')
                # adv.utils.print_network(state.opt.discriminator.subdiscriminators[0])

                state.opt.diff_loss = adv.customized_loss.DiffLoss()

            # Init the fair supervised contrastive loss
            if state.FCL:
                state.opt.FairSCL = FairCL.Fair_Contrastive_Loss(state)
            
            # Init the group difference loss
            if (state.DyBT is not None) and (state.DyBT == "GroupDifference"):
                state.opt.group_difference_loss = Group_Difference_Loss(state)
            
            # Init the UKNN loss for unsupervised bias mitigation
            if state.knn_labels and state.UKNN_debiasing and (state.UKNN_lambda != 0):
                state.opt.UKNN_loss = KNN_Loss(state)

            # Init the ARL for unsupervised training
            if state.ARL:
                assert not state.adv_debiasing, "ARL is unsupervised bias mitigation, which cannot be used together with adversarial training"
                state.opt.ARL_loss = ARL(state)

        return state


# options = BaseOptions()
