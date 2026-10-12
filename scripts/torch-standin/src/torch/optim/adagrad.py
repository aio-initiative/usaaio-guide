# Ported from PyTorch (torch/optim/adagrad.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.optim.Adagrad, with torch's lr_decay and initial accumulator."""

import numpy as np

from .._tensor import Tensor
from .optimizer import Optimizer, quiet, updated


class Adagrad(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach=None,
        *,
        maximize=False,
        differentiable=False,
        fused=None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(f"Invalid initial_accumulator_value value: {initial_accumulator_value}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
            "fused": fused,
        }
        super().__init__(params, defaults)
        # Unlike the other optimisers, Adagrad makes every parameter's state when it is built.
        for group in self.param_groups:
            for p in group["params"]:
                self._start(p)

    def _start(self, p):
        state = self.state[p]
        state["step"] = Tensor(np.zeros((), dtype=np.float32))
        state["sum"] = Tensor(np.full_like(p._data, self.defaults["initial_accumulator_value"]))

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with quiet():
            for group in self.param_groups:
                self._update(group)
        return loss

    def _update(self, group):
        lr, lr_decay, eps = float(group["lr"]), group["lr_decay"], group["eps"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if len(self.state[p]) == 0:  # a parameter added with add_param_group
                self._start(p)
            state = self.state[p]
            state["step"]._data += 1
            step = float(state["step"]._data)
            grad = p.grad._data
            if group["maximize"]:
                grad = -grad
            if weight_decay != 0:
                grad = grad + weight_decay * p._data
            clr = lr / (1 + (step - 1) * lr_decay)
            total = state["sum"]._data
            total += grad * grad
            std = np.sqrt(total) + eps
            p._data += -clr * (grad / std)
            updated(p)
