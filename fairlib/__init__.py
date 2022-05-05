import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# try:
#     from .src.base_options import BaseOptions
#     from .src import analysis
#     from .src import dataloaders
#     from .src import evaluators
#     from .src import networks
#     from .src import utils
# except:
#     from src.base_options import BaseOptions
#     from src import analysis
#     from src import dataloaders
#     from src import evaluators
#     from src import networks
#     from src import utils

from fairlib.src.base_options import BaseOptions
from fairlib.src import analysis
from fairlib.src import dataloaders
from fairlib.src import evaluators
from fairlib.src import networks
from fairlib.src import utils