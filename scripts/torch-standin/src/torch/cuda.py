"""torch.cuda: there is no GPU in the browser."""


def is_available():
    return False


def device_count():
    return 0
