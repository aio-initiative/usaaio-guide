# Ported from PyTorch (torch/optim/adam.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.optim.Adam and AdamW, with torch's bias correction and eps placement."""

import numpy as np

from .._tensor import Tensor
from .optimizer import Optimizer, lerp_, quiet, updated


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        decoupled_weight_decay=False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (isinstance(betas[0], float) and isinstance(betas[1], float)):
            raise ValueError("betas must be either both floats or both Tensors")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
            "decoupled_weight_decay": decoupled_weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = self._evaluate(closure)
        with quiet():
            for group in self.param_groups:
                self._update(group)
        return loss

    def _update(self, group):
        lr, eps, weight_decay = float(group["lr"]), group["eps"], group["weight_decay"]
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = Tensor(np.zeros((), dtype=np.float32))
                state["exp_avg"] = Tensor(np.zeros_like(p._data))
                state["exp_avg_sq"] = Tensor(np.zeros_like(p._data))
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = Tensor(np.zeros_like(p._data))
            state["step"]._data += 1
            step = float(state["step"]._data)
            grad = p.grad._data
            if group["maximize"]:
                grad = -grad
            if weight_decay != 0:
                if group["decoupled_weight_decay"]:
                    p._data *= 1 - lr * weight_decay
                else:
                    grad = grad + weight_decay * p._data
            exp_avg, exp_avg_sq = state["exp_avg"]._data, state["exp_avg_sq"]._data
            lerp_(exp_avg, grad, 1 - beta1)
            exp_avg_sq *= beta2
            exp_avg_sq += (grad * grad) * (1 - beta2)
            bias_correction1 = 1 - beta1**step
            bias_correction2_sqrt = (1 - beta2**step) ** 0.5
            if group["amsgrad"]:
                longest = state["max_exp_avg_sq"]._data
                np.maximum(longest, exp_avg_sq, out=longest)
                denom = np.sqrt(longest) / bias_correction2_sqrt + eps
            else:
                denom = np.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
            p._data += (-lr / bias_correction1) * (exp_avg / denom)
            updated(p)


class AdamW(Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )
