def torch_rowwise_sum_baseline(x):
    """Framework eager baseline; timing must say whether dispatch is included."""
    import torch

    if not x.is_cuda:
        raise ValueError("expected a GPU tensor")
    if x.dtype is not torch.float32:
        raise ValueError("expected float32")
    return torch.sum(x, dim=1)


def oneflow_rowwise_sum_baseline(x):
    """OneFlow eager baseline sketch with the same rowwise contract."""
    import oneflow as flow

    if not x.is_cuda:
        raise ValueError("expected a GPU tensor")
    if x.dtype is not flow.float32:
        raise ValueError("expected float32")
    return flow.sum(x, dim=1)
