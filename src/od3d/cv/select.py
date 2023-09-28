import torch

# gather: input, dim, index -> return tensor with source indices (on right side)
#   -> index = source-index
#   -> number of dimensions for input, and index must be the same
#   -> out will have same shape as index
#   -> useful for index2value

# scatter: input, dim, index, src -> return tensor with target indices (on left side)
#   -> index = target-index
#   -> out will have same shape as input, with override
#   -> backward pass only works for src.shape == index.shape
#   -> useful for index2mask

# masked_scatter: input, mask, source -> return tensor with masked copy of source into target
#   -> mask must be broadcastable for source and target


def batched_index_select(input, index, dim=None):
    """
    Args:
        input: B...x...xIx...
        dim (int): dimension of input which should be indexed (I)
        index: B...xN
    Returns:
        out: B...xNx...
    """
    if dim is None:
        dim = index.dim()-1

    batch_dims = index.shape[:-1]
    batch_dims_count = len(batch_dims)

    views = batch_dims + torch.Size([1 if i != dim else -1 for i in range(batch_dims_count, input.dim())])
    index = index.view(views)

    expanse = batch_dims + torch.Size([input.shape[i] if i != dim else -1 for i in range(batch_dims_count, input.dim())]) #
    index = index.expand(expanse)

    return torch.gather(input, dim, index)



