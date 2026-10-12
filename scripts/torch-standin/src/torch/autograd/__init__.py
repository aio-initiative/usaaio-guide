"""torch.autograd: grad and backward, the gradient modes, and autograd.functional."""

from .._autograd import backward, enable_grad, grad, no_grad, set_grad_enabled
from .._missing import module_getattr
from . import functional

__getattr__ = module_getattr(__name__, "torch.autograd")
