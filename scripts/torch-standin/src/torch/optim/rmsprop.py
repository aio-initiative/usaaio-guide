# Ported from PyTorch (torch/optim/rmsprop.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.optim.RMSprop, with torch's momentum, centered average and eps placement."""

import numpy as np

from .._tensor import Tensor
from .optimizer import Optimizer, lerp_, quiet, updated


class RMSprop(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        capturable=False,
        foreach=None,
        maximize=False,
        differentiable=False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "alpha": alpha,
            "eps": eps,
            "centered": centered,
            "weight_decay": weight_decay,
            "capturable": capturable,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with quiet():
            for group in self.param_groups:
                self._update(group)
        return loss

    def _update(self, group):
        lr, alpha, eps = float(group["lr"]), group["alpha"], group["eps"]
        momentum, weight_decay = group["momentum"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = Tensor(np.zeros((), dtype=np.float32))
                state["square_avg"] = Tensor(np.zeros_like(p._data))
                if momentum > 0:
                    state["momentum_buffer"] = Tensor(np.zeros_like(p._data))
                if group["centered"]:
                    state["grad_avg"] = Tensor(np.zeros_like(p._data))
            state["step"]._data += 1
            grad = p.grad._data
            if group["maximize"]:
                grad = -grad
            if weight_decay != 0:
                grad = grad + weight_decay * p._data
            square_avg = state["square_avg"]._data
            square_avg *= alpha
            square_avg += (grad * grad) * (1 - alpha)
            if group["centered"]:
                grad_avg = state["grad_avg"]._data
                lerp_(grad_avg, grad, 1 - alpha)
                avg = np.sqrt(square_avg - grad_avg * grad_avg) + eps
            else:
                avg = np.sqrt(square_avg) + eps
            if momentum > 0:
                buf = state["momentum_buffer"]._data
                buf *= momentum
                buf += grad / avg
                p._data += -lr * buf
            else:
                p._data += -lr * (grad / avg)
            updated(p)
