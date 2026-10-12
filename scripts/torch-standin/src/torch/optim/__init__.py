"""torch.optim: the SGD, RMSprop, Adagrad, Adam and AdamW optimisers and torch.optim.lr_scheduler."""

from .._missing import module_getattr
from . import lr_scheduler
from .adagrad import Adagrad
from .adam import Adam, AdamW
from .optimizer import Optimizer
from .rmsprop import RMSprop
from .sgd import SGD

__getattr__ = module_getattr(__name__, "torch.optim")
