# Ported from PyTorch (torch/nn/modules/loss.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""The loss layers: each calls its function in torch.nn.functional with its settings. Class
weights (and BCEWithLogitsLoss's pos_weight) are buffers, so they are in state_dict()."""

from .. import functional as F
from .module import Module


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__()
        if size_average is not None or reduce is not None:
            reduction = F._reduction(size_average, reduce, reduction)
        self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)


class MSELoss(_Loss):
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)


class L1Loss(_Loss):
    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)


class HuberLoss(_Loss):
    def __init__(self, reduction="mean", delta=1.0):
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class CrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index, self.label_smoothing = ignore_index, label_smoothing

    def forward(self, input, target):
        return F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class NLLLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class BCELoss(_WeightedLoss):
    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None):
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(
            input, target, self.weight, pos_weight=self.pos_weight, reduction=self.reduction
        )
