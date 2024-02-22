import torch
from transformers import GPT2LMHeadModel, OPTForCausalLM, FalconForCausalLM, OpenLlamaForCausalLM, OPTModel


def flatten_input_embeds(input_embeds):
    return input_embeds.reshape(input_embeds.shape[0], -1)


def unflatten_input_embeds(input_embeds, input_shape):
    return input_embeds.reshape(input_embeds.shape[0], *input_shape)

def get_embeds(model, input_ids):
    if isinstance(model, GPT2LMHeadModel):
        inputs_embeds = model.transformer.wte(input_ids)
    elif isinstance(model, OPTForCausalLM):
        inputs_embeds = model.model.decoder.embed_tokens(input_ids)
    elif isinstance(model, FalconForCausalLM):
        inputs_embeds = model.transformer.word_embeddings(input_ids)
    elif isinstance(model, OpenLlamaForCausalLM):
        inputs_embeds = model.model.embed_tokens(input_ids)
    elif isinstance(model, OPTModel):
        inputs_embeds =  model.model.decoder.embed_tokens(input_ids)
    else:
        raise ValueError(f"model {model} not supported")
    return inputs_embeds
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