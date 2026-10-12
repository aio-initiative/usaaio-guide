"""torch.nn: layers, losses and containers on the stand-in's tensors.

Every import here is relative, as in the rest of the package. A torch.nn name the stand-in
does not have gives the not-available message.
"""

from .._missing import module_getattr as _module_getattr
from . import functional, init
from .modules import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    BCELoss,
    BCEWithLogitsLoss,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    CrossEntropyLoss,
    Dropout,
    ELU,
    Embedding,
    Flatten,
    GELU,
    GroupNorm,
    HuberLoss,
    Identity,
    L1Loss,
    LayerNorm,
    LeakyReLU,
    Linear,
    MSELoss,
    MaxPool2d,
    Module,
    ModuleDict,
    ModuleList,
    MultiheadAttention,
    NLLLoss,
    ReLU,
    Sequential,
    SiLU,
    Sigmoid,
    Softmax,
    Tanh,
    Upsample,
)
from .parameter import Parameter

__getattr__ = _module_getattr(__name__, "torch.nn")
