from copy import deepcopy
import torch
from ...data.data_types import GenBatch, LatentDataConfig

try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    print("Using slow linear assignment implementation")
    from .utils import batch_linear_assignment


def get_batch_assignment(
    reference: GenBatch,
    target: GenBatch,
    data_conf: LatentDataConfig,
    global_denoms: dict[str, torch.Tensor] | None = None,
):
    assert reference.shape == target.shape
    assert (reference.lengths == target.lengths).all()
    L, B = reference.shape

    cost = torch.zeros((B, L, L), device=reference.device)

    # 1. Compute cat
    for name in data_conf.focus_cat:
        pred = reference[name].T  # B, L
        true = target[name].T  # B, L
        cost += (pred[:, :, None] != true[:, None, :]) / L  # B, L, L

    # 2. Compute num
    for name in data_conf.focus_num:
        if name == data_conf.time_name:
            pred = reference.time.T
            true = target.time.T.to(dtype=torch.float32)
        else:
            pred = reference[name].T
            true = target[name].T.to(dtype=torch.float32)
        if global_denoms is not None:
            denominator = global_denoms[name]
        else:
            denominator = torch.abs(true - torch.median(true, 1)[0][:, None])  # B, L
            denominator = denominator.sum(axis=1).mean()
        nominator = torch.abs(pred[:, :, None] - true[:, None, :])  # [B, L, L]
        cost += nominator / denominator

    if cost.isnan().any():
        return torch.tensor(float("nan"))
    assignment = batch_linear_assignment(cost)  # B, L
    return assignment, cost


def match_batches(reference: GenBatch, target: GenBatch, data_conf: LatentDataConfig):
    assignment = get_batch_assignment(reference, target, data_conf)[0].T  # L, B

    for attr in ["time", "num_features", "cat_features"]:
        field = getattr(target, attr)
        if isinstance(field, torch.Tensor):
            assign = assignment.unsqueeze(-1) if field.ndim > 2 else assignment
            field = field.take_along_dim(assign, 0)
        setattr(target, attr, field)
    return target


def reverse_with_hist(batch: GenBatch, hist: GenBatch, batch_transforms, data_conf):
    if not batch_transforms:
        return batch
    hist = deepcopy(hist)
    hist.target_time = batch.time
    hist.target_num_features = batch.num_features
    hist.target_cat_features = batch.cat_features

    batch = hist
    for tf in reversed(batch_transforms):
        tf.reverse(batch)
    batch = batch.tail(data_conf.generation_len)
    return batch


def get_best_cost(
    reference: GenBatch,
    target: GenBatch,
    data_conf: LatentDataConfig,
    batch_transforms=None,
    orig_hist: GenBatch = None,
    global_denoms: dict[str, torch.Tensor] | None = None,
):
    if batch_transforms:
        reference = reverse_with_hist(reference, orig_hist, batch_transforms, data_conf)
        target = reverse_with_hist(target, orig_hist, batch_transforms, data_conf)
    result = get_batch_assignment(reference, target, data_conf, global_denoms)
    if not isinstance(result, tuple):
        return result  # NaN tensor
    assignment, cost = result
    cost = cost.take_along_dim(assignment.unsqueeze(-1), -1)  # B, L, 1
    cost = cost.sum((1, 2))  # B
    return cost


def post_process_generation(pred: GenBatch, hist: GenBatch, linear_clip=False):
    if not hist.monotonic_time:
        pred.time = pred.time.clip(min=0)
        return pred
    else:
        order = pred.time.argsort(dim=0)  # (L, B).
        for attr in ["time"]: # , "num_features", "cat_features"]:
            tensor = getattr(pred, attr)
            if tensor is None:
                continue
            shaped_order = order.reshape(
                *(list(order.shape) + [1] * (tensor.ndim - order.ndim))
            )
            tensor = tensor.take_along_dim(shaped_order, dim=0)
            setattr(pred, attr, tensor)
        # Clipping
        last_time = hist.tail(1).time  # 1, B
        pred.time[0] = torch.maximum(pred.time[0], last_time.squeeze(0))
        pred.time = torch.cummax(pred.time, dim=0).values
        return pred

        # if not linear_clip:  # Simple Clip
        #     mask = pred.time < last_time
        #     pred.time[mask] = last_time.expand(pred.time.shape)[mask]
        # else:
        #     L, B = pred.time.shape
        #     time = pred.time
        #     lt = last_time.expand(L, B)
        #     k = (time <= lt).sum(dim=0)  # Number of bad times
        #     i = torch.arange(L, device=time.device).view(L, 1)
        #     mask = i < k
        #     t_end = time.gather(0, k.clamp(max=L - 1).view(1, B))  # First good time
        #     t_end = torch.where(k == L, last_time, t_end)  # If all times bad
        #     pred.time = torch.where(
        #         mask,
        #         last_time
        #         + ((i + 1).float() / (k.clamp(min=1) + 1)) * (t_end - last_time),
        #         time,
        #     )  # Linear approximation from last time to first greater time.

