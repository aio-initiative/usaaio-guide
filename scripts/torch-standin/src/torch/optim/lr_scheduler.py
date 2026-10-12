# Ported from PyTorch (torch/optim/lr_scheduler.py), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""torch.optim.lr_scheduler: LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
LinearLR, OneCycleLR and ReduceLROnPlateau.

Each follows torch's chainable form: get_lr() works out the next rate from the optimiser's
current one, so a rate changed by hand between steps carries on from where it was, as in torch.
The arithmetic is plain Python floats, so the rates equal torch's exactly.
"""

import math
import types
from collections import Counter

from .._missing import module_getattr
from .optimizer import Optimizer


def _per_group(name, optimizer, value):
    """`value` as a list with one entry per parameter group (a list or tuple must already be one)."""
    if isinstance(value, (list, tuple)):
        if len(value) != len(optimizer.param_groups):
            raise ValueError(
                f"{name} must have the same length as optimizer.param_groups. "
                f"{name} has {len(value)} values, param_groups has {len(optimizer.param_groups)}."
            )
        return list(value)
    return [value] * len(optimizer.param_groups)


class LRScheduler:
    """Sets each parameter group's lr for epoch 0 when it is made; step() moves it on one epoch.
    Call optimizer.step() before scheduler.step()."""

    _is_initial = False

    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        f"param 'initial_lr' is not specified in param_groups[{i}] when resuming scheduler with "
                        "last_epoch >= 0.\nThis typically happens when:\n"
                        "1. You're trying to resume training from a checkpoint but haven't properly loaded the "
                        "optimizer state\n"
                        "2. You're using last_epoch >= 0 for a fresh training run (not recommended)"
                    )
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._step_count = 0
        self._is_initial = True
        try:
            self.step()
        finally:
            self._is_initial = False

    def state_dict(self):
        """Everything but the optimiser."""
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """The rates the last step() set, one per parameter group."""
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self._step_count += 1
        self.last_epoch += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _current(self):
        return [group["lr"] for group in self.optimizer.param_groups]


_LRScheduler = LRScheduler  # what older notebooks subclass


class LambdaLR(LRScheduler):
    """lr = initial lr * lr_lambda(epoch), with one function for all groups or one each."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        """As torch's: plain functions are not saved, only the state of callable objects."""
        state = {k: v for k, v in self.__dict__.items() if k not in ("optimizer", "lr_lambdas")}
        state["lr_lambdas"] = [
            None if isinstance(fn, types.FunctionType) else fn.__dict__.copy() for fn in self.lr_lambdas
        ]
        return state

    def load_state_dict(self, state_dict):
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        state_dict["lr_lambdas"] = lr_lambdas
        for fn, saved in zip(self.lr_lambdas, lr_lambdas):
            if saved is not None:
                fn.__dict__.update(saved)

    def get_lr(self):
        return [base * fn(self.last_epoch) for fn, base in zip(self.lr_lambdas, self.base_lrs)]


class StepLR(LRScheduler):
    """Multiplies the lr by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self._current()
        return [lr * self.gamma for lr in self._current()]


class MultiStepLR(LRScheduler):
    """Multiplies the lr by gamma at each epoch in `milestones` (a repeated milestone counts twice)."""

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return self._current()
        return [lr * self.gamma ** self.milestones[self.last_epoch] for lr in self._current()]


class LinearLR(LRScheduler):
    """Scales the lr linearly from start_factor to end_factor over total_iters epochs, then holds."""

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError("Starting multiplicative factor expected to be greater than 0 and less or equal to 1.")
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [lr * self.start_factor for lr in self._current()]
        if self._is_initial or self.last_epoch > self.total_iters:
            return self._current()
        change = self.end_factor - self.start_factor
        return [
            lr * (1.0 + change / (self.total_iters * self.start_factor + (self.last_epoch - 1) * change))
            for lr in self._current()
        ]


class ExponentialLR(LRScheduler):
    """Multiplies the lr by gamma every epoch."""

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._is_initial:
            return self._current()
        return [lr * self.gamma for lr in self._current()]


class CosineAnnealingLR(LRScheduler):
    """Follows half a cosine from the initial lr down to eta_min over T_max epochs, then back up."""

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._is_initial:
            return self._current()
        if self._step_count == 1 and self.last_epoch > 0:  # resumed in the middle of the curve
            return [
                self.eta_min + (base - self.eta_min) * (1 + math.cos(self.last_epoch * math.pi / self.T_max)) / 2
                for base in self.base_lrs
            ]
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                lr + (base - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base, lr in zip(self.base_lrs, self._current())
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (lr - self.eta_min)
            + self.eta_min
            for lr in self._current()
        ]


class OneCycleLR(LRScheduler):
    """One cycle: the lr rises from max_lr / div_factor to max_lr over the first pct_start of
    total_steps, then falls to initial_lr / final_div_factor, while the optimiser's momentum (or
    Adam's beta1) moves the other way. Call step() after every batch, not every epoch."""

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(f"Expected positive integer total_steps, but got {total_steps}")
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is not None:
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
                raise ValueError(f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}")
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        # Each phase runs to end_step, from the start_ to the end_ value of the lr and momentum.
        if three_phase:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": float(2 * pct_start * self.total_steps) - 2,
                    "start_lr": "max_lr",
                    "end_lr": "initial_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "max_momentum",
                },
            ]
        else:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": "max_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
            ]
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(f"Expected float between 0 and 1 pct_start, but got {pct_start}")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"anneal_strategy must be one of 'cos' or 'linear', instead got {anneal_strategy}")
        self._anneal_func_type = anneal_strategy
        max_lrs = _per_group("max_lr", optimizer, max_lr)
        if last_epoch == -1:
            for group, top in zip(optimizer.param_groups, max_lrs):
                group["initial_lr"] = top / div_factor
                group["max_lr"] = top
                group["min_lr"] = group["initial_lr"] / final_div_factor
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if "momentum" not in optimizer.defaults and "betas" not in optimizer.defaults:
                raise ValueError("optimizer must support momentum or beta1 with `cycle_momentum` option enabled")
            self.use_beta1 = "betas" in optimizer.defaults
            max_momentums = _per_group("max_momentum", optimizer, max_momentum)
            base_momentums = _per_group("base_momentum", optimizer, base_momentum)
            if last_epoch == -1:
                for top, bottom, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        group["betas"] = (top, *group["betas"][1:])
                    else:
                        group["momentum"] = top
                    group["max_momentum"] = top
                    group["base_momentum"] = bottom
        super().__init__(optimizer, last_epoch)

    def _anneal(self, start, end, pct):
        if self._anneal_func_type == "cos":
            return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)
        return (end - start) * pct + start

    def get_lr(self):
        lrs = []
        step_num = self.last_epoch
        if step_num > self.total_steps:
            raise ValueError(f"Tried to step {step_num} times. The specified number of total steps is {self.total_steps}")
        for group in self.optimizer.param_groups:
            start_step = 0.0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    lr = self._anneal(group[phase["start_lr"]], group[phase["end_lr"]], pct)
                    if self.cycle_momentum:
                        momentum = self._anneal(group[phase["start_momentum"]], group[phase["end_momentum"]], pct)
                    break
                start_step = phase["end_step"]
            lrs.append(lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    group["betas"] = (momentum, *group["betas"][1:])
                else:
                    group["momentum"] = momentum
        return lrs


class ReduceLROnPlateau(LRScheduler):
    """Cuts the lr by `factor` when the metric given to step(metric) has not improved for more than
    `patience` epochs, then waits `cooldown` epochs before counting again."""

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.default_min_lr = None
            self.min_lrs = list(min_lr)
        else:
            self.default_min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = self._current()
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # bad epochs during the cooldown do not count
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        self._last_lr = self._current()

    def _reduce_lr(self):
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError(
                    "The number of param groups in the `optimizer` "
                    f"({len(self.optimizer.param_groups)}) differs "
                    f"from when `ReduceLROnPlateau` was initialized "
                    f"({len(self.min_lrs)}), usually due to a new "
                    "param group being added to the optimizer. Please "
                    "modify the `min_lrs` field to match the length "
                    "of the `optimizer` param groups."
                )
            self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)
        for group, min_lr in zip(self.optimizer.param_groups, self.min_lrs):
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, min_lr)
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            return a < best * (1.0 - self.threshold)
        if self.mode == "min":
            return a < best - self.threshold
        if self.threshold_mode == "rel":
            return a > best * (self.threshold + 1.0)
        return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in ("min", "max"):
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        self.mode_worse = math.inf if mode == "min" else -math.inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)


__getattr__ = module_getattr(__name__, "torch.optim.lr_scheduler")
