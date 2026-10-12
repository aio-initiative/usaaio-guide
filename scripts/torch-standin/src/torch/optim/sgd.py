# Ported from PyTorch (torch/optim/sgd.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.optim.SGD, with torch's momentum, dampening, Nesterov and weight decay."""

import numpy as np

from .._tensor import Tensor
from .optimizer import Optimizer, quiet, updated


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
            "foreach": foreach,
            "differentiable": differentiable,
            "fused": fused,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with quiet():
            for group in self.param_groups:
                self._update(group)
        return loss

    def _update(self, group):
        lr = float(group["lr"])
        momentum, dampening = group["momentum"], group["dampening"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad._data
            if group["maximize"]:
                grad = -grad
            if weight_decay != 0:
                grad = grad + weight_decay * p._data
            if momentum != 0:
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    # The first buffer is the gradient itself, not scaled by dampening.
                    buf = state["momentum_buffer"] = Tensor(np.array(grad))
                else:
                    buf._data *= momentum
                    buf._data += (1 - dampening) * grad
                grad = grad + momentum * buf._data if group["nesterov"] else buf._data
            p._data += -lr * grad
            updated(p)
