from copy import deepcopy

import torch

from ..models.generator.utils import get_best_cost


def compute_global_denoms(batch_input, batch_transforms, data_conf):
    """Per-feature global denominator matching OTD's r1_score normalisation.

    denom = |target - mode_baseline|.sum(L).mean(B)  (scalar per feature).

    Reverses *batch_input* into original data space; after
    ``CutTargetSequence.reverse`` the full sequence lives in
    ``.time`` / ``.num_features``, and the target is the last
    ``generation_len`` items.

    Returns dict[name] -> scalar tensor.
    """
    gen_len = data_conf.generation_len
    max_seq_len = data_conf.max_seq_len

    eff = deepcopy(batch_input)
    if batch_transforms:
        for tf in reversed(batch_transforms):
            tf.reverse(eff)
    else:
        # No reverse pipeline: build full sequence explicitly as hist + target.
        eff.append(eff.get_target_batch())

    device = eff.time.device
    T, B = eff.time.shape
    lengths = eff.lengths
    pos = torch.arange(T, device=device).unsqueeze(1)          # (T, 1)

    hist_end = lengths - gen_len                               # (B,) exclusive
    target_offsets = torch.arange(gen_len, device=device).unsqueeze(1)  # (gen_len, 1)
    target_idx = hist_end.unsqueeze(0) + target_offsets        # (gen_len, B)

    hist_window = max_seq_len - gen_len
    hist_start = (hist_end - hist_window).clamp(min=0)     # (B,)
    hist_mask = (pos >= hist_start.unsqueeze(0)) & (pos < hist_end.unsqueeze(0))

    denoms = {}
    for name in data_conf.focus_num:
        if name == data_conf.time_name:
            full = eff.time.float()                            # (T, B)
        else:
            idx = eff.num_features_names.index(name)
            full = eff.num_features[..., idx].float()          # (T, B)

        target = full.gather(0, target_idx)                    # (gen_len, B)
        hist_masked = full.masked_fill(~hist_mask, float("nan"))  # (T, B)

        if name == data_conf.time_name:
            hist_diff = hist_masked.diff(dim=0)                # NaN-propagating diff
            median_delta = torch.nanquantile(hist_diff, 0.5, dim=0)  # (B,)
            last_hist_idx = (hist_end - 1).clamp(min=0)
            last_hist = full.gather(0, last_hist_idx.unsqueeze(0)).squeeze(0)  # (B,)
            arange = torch.arange(
                1, gen_len + 1, device=device, dtype=torch.float32,
            ).unsqueeze(1)                                     # (gen_len, 1)
            baseline = last_hist.unsqueeze(0) + arange * median_delta.unsqueeze(0)
        else:
            baseline = torch.nanquantile(hist_masked, 0.5, dim=0)  # (B,)
            baseline = baseline.unsqueeze(0).expand(gen_len, -1)
        denoms[name] = (target - baseline).abs().sum(0).mean().clamp(min=1e-8)
    return denoms


def pairwise_medoid_indices(
    sampled_batches,
    batch_input,
    data_conf,
    batch_transforms=None,
    global_denoms=None,
):
    """For each user, pick the medoid sample: the one minimising the sum of
    transport costs to all other samples. Returns (B,) long tensor of
    indices into sampled_batches.
    """
    N = len(sampled_batches)
    B = batch_input.shape[1]
    total_distances = torch.zeros(N, B, device=batch_input.device)
    for i in range(N):
        for j in range(i + 1, N):
            dist_ij = get_best_cost(
                deepcopy(sampled_batches[i]),
                deepcopy(sampled_batches[j]),
                data_conf,
                batch_transforms=batch_transforms,
                orig_hist=deepcopy(batch_input),
                global_denoms=global_denoms,
            )
            total_distances[i] += dist_ij
            total_distances[j] += dist_ij
    return total_distances.argmin(dim=0)


def select_medoid(sampled_batches, indices):
    """Build a GenBatch by picking per-user sample from *sampled_batches*.

    indices: (B,) long tensor — index into the N samples for each user.
    """
    x_hat = deepcopy(sampled_batches[0])
    for attr in ["time", "num_features", "cat_features"]:
        field = getattr(x_hat, attr)
        if not isinstance(field, torch.Tensor):
            continue
        stacked = torch.stack(
            [getattr(sb, attr) for sb in sampled_batches], dim=-1,
        )
        if field.ndim == 3:
            idx = indices[None, :, None, None].expand(*stacked.shape[:-1], 1)
        else:
            idx = indices[None, :, None].expand(*stacked.shape[:-1], 1)
        setattr(x_hat, attr, torch.gather(stacked, -1, idx).squeeze(-1))
    return x_hat
