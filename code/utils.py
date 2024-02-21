import torch


def to_cpu(obj):
    """
    Recursively moves tensors in a dictionary or in any nested structure
    to CPU. If `obj` is a dictionary, it processes its values. If `obj` is
    an iterable, it processes its items. If `obj` is a tensor, it moves
    it to CPU.

    Parameters:
    - obj: The object to process. Can be a dictionary, an iterable, or a tensor.

    Returns:
    - The same structure with all tensors moved to CPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, str):
        # For lists, tuples, and other iterables, except for strings
        return type(obj)(to_cpu(v) for v in obj)
    else:
        return obj