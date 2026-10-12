"""The layer classes of torch.nn, one family per file as in torch."""

from .activation import ELU, GELU, LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Tanh
from .attention import MultiheadAttention
from .container import ModuleDict, ModuleList, Sequential
from .conv import Conv2d, ConvTranspose2d
from .dropout import Dropout
from .flatten import Flatten
from .linear import Identity, Linear, NonDynamicallyQuantizableLinear
from .loss import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, HuberLoss, L1Loss, MSELoss, NLLLoss
from .module import Module
from .normalization import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm
from .pooling import AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from .sparse import Embedding
from .upsampling import Upsample
