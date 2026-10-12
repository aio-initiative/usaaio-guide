"""torch.distributions: the normal distribution and its KL divergence."""

from .._missing import module_getattr
from .kl import kl_divergence
from .normal import Normal

__getattr__ = module_getattr(__name__, "torch.distributions")
