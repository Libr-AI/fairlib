try:
    from .src.base_options import BaseOptions
    from .src import analysis
    from .src import dataloaders
    from .src import evaluators
    from .src import networks
    from .src import utils
except:
    from src.base_options import BaseOptions
    from src import analysis
    from src import dataloaders
    from src import evaluators
    from src import networks
    from src import utils