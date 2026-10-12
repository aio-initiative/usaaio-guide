"""torch.distributions.kl_divergence, for the pairs of distributions the stand-in has."""

from .normal import Normal


def kl_divergence(p, q):
    """KL(p || q), elementwise over the batch."""
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    raise NotImplementedError(f"No KL(p || q) is implemented for {type(p).__name__} and {type(q).__name__}")
