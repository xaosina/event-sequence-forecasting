import logging
import time
from contextlib import contextmanager
from copy import deepcopy

import torch
from dacite import Config, from_dict

from ..data.data_types import GenBatch
from ..models.generator import BaseGenerator, ModelConfig
from ..models.generator.utils import (
    get_batch_assignment,
    get_best_cost,
    post_process_generation,
    reverse_with_hist,
)
from ..utils import freeze_module
from .booster_utils import (
    compute_global_denoms,
    pairwise_medoid_indices,
    select_medoid,
)

logger = logging.getLogger(__name__)


def get_device(module):
    for param in module.parameters():
        return param.device
    return None


class Booster:
    def __init__(
        self,
        data_conf,
        n_samples: int = 1,
        stategy: str = "default",
        focus_on: list[str] = None,
        params: dict = None,
    ):
        if focus_on:
            data_conf = deepcopy(data_conf)
            object.__setattr__(data_conf, "focus_on", focus_on)
        self.data_conf = data_conf
        self.stategy = stategy
        self.n_samples = n_samples
        self.params = params or {}
        if params is None:
            self.reference = None
        else:
            self.reference = self._init_reference(params.get("reference"))
        self._profile = (
            dict(total=0.0, sampling=0.0, medoid=0.0, matching=0.0,
                 aggregation=0.0, postprocess=0.0, n_generate=0, n_batches=0)
            if self.params.get("profile") else None
        )

    @contextmanager
    def _timed(self, stage: str):
        """Accumulate wall-time of a stage into ``self._profile`` (no-op unless
        ``params.profile``). Syncs CUDA so async kernels are actually measured."""
        if self._profile is None:
            yield
            return
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            self._profile[stage] += time.perf_counter() - t0

    def _init_reference(self, params):
        if params is None:
            return None
        checkpoint = params.pop("checkpoint", None)
        cfg = from_dict(ModelConfig, params, Config(strict=True))
        reference_generator = BaseGenerator.get_model(
            params["name"], self.data_conf, cfg
        )
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = reference_generator.load_state_dict(ckpt["model"], strict=True)
            print("Reference booster:", msg)
            reference_generator = freeze_module(reference_generator)
        return reference_generator

    def __call__(
        self,
        model,
        batch_input,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        with self._timed("total"):
            return getattr(self, self.stategy)(
                model, batch_input, gen_len, batch_transforms, topk, temperature
            )

    def default(
        self,
        model,
        batch_input,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        with torch.no_grad():
            batch_pred = model.generate(
                batch_input,
                gen_len,
                topk=topk,
                temperature=temperature,
            )
        return batch_pred

    def perfect(
        self,
        model,
        batch_input: GenBatch,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        # assert batch_input.monotonic_time
        matching = self.params.get("matching", "local")
        use_original_space = self.params.get("use_original_space", True)
        cost_tfs = batch_transforms if use_original_space else None
        g_denoms = (
            compute_global_denoms(batch_input, cost_tfs, self.data_conf)
            if matching == "global"
            else None
        )

        best_score = None
        for _ in range(self.n_samples):
            sampled_batch = model.generate(
                deepcopy(batch_input),
                gen_len,
                topk=topk,
                temperature=temperature,
            )
            score = get_best_cost(
                deepcopy(sampled_batch),
                deepcopy(batch_input.get_target_batch()),
                self.data_conf,
                batch_transforms=cost_tfs,
                orig_hist=deepcopy(batch_input),
                global_denoms=g_denoms,
            )  # B
            if best_score is None:
                best_score = score
                pred_batch = sampled_batch
                continue
            improved = score < best_score
            for attr in ["time", "num_features", "cat_features"]:
                field = getattr(pred_batch, attr)
                new_field = getattr(sampled_batch, attr)
                if isinstance(field, torch.Tensor):
                    field[:, improved] = new_field[:, improved]
                setattr(pred_batch, attr, field)
            best_score = torch.where(improved, score, best_score)
        return pred_batch

    def closest_to_reference(
        self,
        model,
        batch_input: GenBatch,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        assert batch_input.monotonic_time
        use_original_space = self.params.get("use_original_space", True)
        cost_tfs = batch_transforms if use_original_space else None
        best_score = None
        if batch_input.device != get_device(self.reference):
            self.reference.to(batch_input.device)
        reference = self.reference.generate(
            deepcopy(batch_input), gen_len, topk=topk, temperature=temperature
        )
        # return reference
        for _ in range(self.n_samples):
            sampled_batch = model.generate(
                deepcopy(batch_input),
                gen_len,
                topk=topk,
                temperature=temperature,
            )
            score = get_best_cost(
                deepcopy(sampled_batch),
                deepcopy(reference),
                self.data_conf,
                batch_transforms=cost_tfs,
                orig_hist=deepcopy(batch_input),
            )  # B
            if best_score is None:
                best_score = score
                pred_batch = sampled_batch
                continue
            improved = score < best_score
            for attr in ["time", "num_features", "cat_features"]:
                field = getattr(pred_batch, attr)
                new_field = getattr(sampled_batch, attr)
                if isinstance(field, torch.Tensor):
                    field[:, improved] = new_field[:, improved]
                setattr(pred_batch, attr, field)
            best_score = torch.where(improved, score, best_score)
        return pred_batch

    def barybooster(
        self,
        model,
        batch_input: GenBatch,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        # assert batch_input.monotonic_time
        matching = self.params.get("matching", "local")
        use_original_space = self.params.get("use_original_space", True)
        cost_tfs = batch_transforms if use_original_space else None
        g_denoms = (
            compute_global_denoms(batch_input, cost_tfs, self.data_conf)
            if matching == "global"
            else None
        )

        sampled_batches = []
        for _ in range(self.n_samples):
            sampled_batch = model.generate(
                deepcopy(batch_input),
                gen_len,
                topk=topk,
                temperature=temperature,
            )
            sampled_batches.append(sampled_batch)

        indices = pairwise_medoid_indices(
            sampled_batches, batch_input, self.data_conf,
            batch_transforms=cost_tfs, global_denoms=g_denoms,
        )
        return select_medoid(sampled_batches, indices)

    def wasserstein_barybooster(
        self,
        model,
        batch_input: GenBatch,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        """Synthesize an optimal batch minimizing sum of transport costs to
        N sampled batches. Alternating optimization:
          - fix x_hat -> compute assignments for each sample (Hungarian);
          - fix assignments -> update x_hat (median for time/num, mode for cat).
        Uses the global (OTD-aligned) matching only.
        """
        N = self.n_samples
        use_original_space = self.params.get("use_original_space", True)
        cost_tfs = batch_transforms if use_original_space else None

        g_denoms = compute_global_denoms(batch_input, cost_tfs, self.data_conf)

        sampled_batches = []
        with self._timed("sampling"):
            for _ in range(N):
                sb = model.generate(
                    deepcopy(batch_input),
                    gen_len,
                    topk=topk,
                    temperature=temperature,
                )
                sampled_batches.append(sb)
        if self._profile is not None:
            self._profile["n_generate"] += N
            self._profile["n_batches"] += 1

        with self._timed("medoid"):
            medoid_idx = pairwise_medoid_indices(
                sampled_batches, batch_input, self.data_conf,
                batch_transforms=cost_tfs, global_denoms=g_denoms,
            )
            x_hat = select_medoid(sampled_batches, medoid_idx)

        max_iter = self.params.get("max_iter", 10)
        verbose = self.params.get("verbose", False)
        post_process_output = self.params.get("post_process_output", False)
        linear_clip = self.params.get("linear_clip", False)

        eff_samples = [
            reverse_with_hist(
                deepcopy(sb), deepcopy(batch_input),
                cost_tfs, self.data_conf,
            )
            for sb in sampled_batches
        ]

        prev_assignments = None
        for iteration in range(max_iter):
            ref = reverse_with_hist(
                deepcopy(x_hat), deepcopy(batch_input),
                cost_tfs, self.data_conf,
            )

            assignments = []
            total_cost = 0.0 if verbose else None
            with self._timed("matching"):
                for eff in eff_samples:
                    result = get_batch_assignment(ref, eff, self.data_conf, g_denoms)
                    if not isinstance(result, tuple):
                        logger.warning("NaN in assignment cost; returning medoid")
                        return x_hat
                    assignment, cost = result
                    assignments.append(assignment)  # B, L
                    if verbose:
                        best = cost.take_along_dim(assignment.unsqueeze(-1), -1).sum((1, 2))
                        total_cost += best.mean().item()
            if verbose:
                logger.info(
                    "[wasserstein] iter %d: mean total cost = %.6f",
                    iteration, total_cost,
                )

            if prev_assignments is not None and all(
                torch.equal(a, p) for a, p in zip(assignments, prev_assignments)
            ):
                logger.info(
                    "Wasserstein barycenter converged at iteration %d", iteration,
                )
                break
            prev_assignments = [a.clone() for a in assignments]

            with self._timed("aggregation"):
                for attr in ["time", "num_features", "cat_features"]:
                    field = getattr(x_hat, attr)
                    if not isinstance(field, torch.Tensor):
                        continue
                    aligned = []
                    for i, sb in enumerate(sampled_batches):
                        src = getattr(sb, attr)
                        assign_T = assignments[i].T  # L, B
                        if src.ndim == 3:
                            assign_T = assign_T.unsqueeze(-1).expand_as(src)
                        aligned.append(src.gather(0, assign_T))
                    stacked = torch.stack(aligned, dim=0)  # N, L, B, [D]
                    if attr == "cat_features":
                        new_field = torch.mode(stacked, dim=0).values
                    else:
                        new_field = stacked.float().median(dim=0).values
                    setattr(x_hat, attr, new_field)

        if post_process_output:
            with self._timed("postprocess"):
                x_hat = post_process_generation(
                    deepcopy(x_hat), deepcopy(batch_input), linear_clip=linear_clip
                )
        return x_hat

    def avg_booster(
        self,
        model,
        batch_input: GenBatch,
        gen_len,
        batch_transforms=None,
        topk=1,
        temperature=1.0,
    ):
        """Dumb barycenter: aggregate N samples with identity matching only."""
        N = self.n_samples
        post_process_output = self.params.get("post_process_output", False)
        linear_clip = self.params.get("linear_clip", False)

        sampled_batches = []
        for _ in range(N):
            sb = model.generate(
                deepcopy(batch_input),
                gen_len,
                topk=topk,
                temperature=temperature,
            )
            sampled_batches.append(sb)

        x_hat = deepcopy(sampled_batches[0])
        for attr in ["time", "num_features", "cat_features"]:
            field = getattr(x_hat, attr)
            if not isinstance(field, torch.Tensor):
                continue
            # Identity matching: keep each sample in its original order.
            stacked = torch.stack([getattr(sb, attr) for sb in sampled_batches], dim=0)
            if attr == "cat_features":
                new_field = torch.mode(stacked, dim=0).values
            else:
                new_field = stacked.float().median(dim=0).values
            setattr(x_hat, attr, new_field)

        if post_process_output:
            x_hat = post_process_generation(
                deepcopy(x_hat), deepcopy(batch_input), linear_clip=linear_clip
            )
        return x_hat
