import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import torch
from dacite import Config, from_dict
from ebes.model import BaseModel, TakeLastHidden
from torch import nn

from ...data.data_types import GenBatch, LatentDataConfig, PredBatch
from generation.utils import freeze_module

from . import BaseGenerator, ModelConfig
from .cdiffu.tabular_diffusion_model import DiffusionTabularModel

logger = logging.getLogger()


@dataclass(frozen=True)
class FeatureLayout:
    num_names: List[str]
    cat_names: List[str]
    num_idxs: List[int]
    cat_idxs: List[int]


class CrossDiffusionModel(BaseGenerator):
    def __init__(self, data_conf: LatentDataConfig, model_config: ModelConfig):
        super().__init__()
        self.data_conf = data_conf
        self.model_config = model_config

        self.help_net, _ = self._init_helpnet()
        self.history_encoder, hist_dim = self._init_history_encoder()

        self.repeat_samples: int = int(model_config.params.get("repeat_samples", 1))
        logger.info(
            f"Generation is reapeating {self.repeat_samples} times to smooth time feature."
        )

        self.history_len: int = int(model_config.params["history_len"])
        self.generation_len: int = int(model_config.params["generation_len"])
        self.model = DiffusionTabularModel(
            data_conf=self.data_conf,
            model_config=self.model_config,
            outer_history_encoder_dim=hist_dim,
            generation_len=self.generation_len,
        )

        self.fix_features: set[str] = set(
            self.model_config.params.get("fix_features", [])
        )
        self.diff_features: set[str] = set(
            self.model_config.params.get("diff_features", [])
        )
        self.focus: set[str] = self.fix_features | self.diff_features
        self.cfg_w = float(self.model_config.params.get("cfg_w", 1.0))
        logger.info(f"Cfg was set as {self.cfg_w}")

    @staticmethod
    def _to_blf(t: torch.Tensor) -> torch.Tensor:
        # (L, B, F) -> (B, L, F) or (L, B) -> (B, L)
        return t.permute(1, 0, *range(2, t.ndim))

    def _init_helpnet(self) -> BaseGenerator:
        params = self.model_config.params.get("help_net")
        if not params:
            print("No HelpNet provided.")
            return None, None

        params = deepcopy(params)
        checkpoint = params.pop("checkpoint", None)
        name = params.get("name")

        if name == "GroundTruthGenerator":
            enc = BaseModel.get_model(name)
            hist_dim = None
        elif name == "GRU":
            input_size = params.get("input_size", 256)
            gru_cfg = dict(input_size=input_size, num_layers=1, hidden_size=256)
            hist_dim = 256
            enc = nn.Sequential(BaseModel.get_model("GRU", **gru_cfg), TakeLastHidden())
        else:
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            enc = BaseGenerator.get_model(name, self.data_conf, cfg)
            hist_dim = getattr(getattr(enc, "encoder", None), "output_dim", None)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = enc.load_state_dict(ckpt["model"], strict=True)
            logger.info("History encoder: " + str(msg))
            enc = freeze_module(enc)

        return enc, hist_dim

    def _init_history_encoder(self) -> BaseGenerator:
        params = self.model_config.params.get("history_encoder")
        if not params:
            print("No history encoder provided.")
            return None, None

        params = deepcopy(params)
        checkpoint = params.pop("checkpoint", None)
        name = params.get("name")

        if name == "AutoregressiveGenerator":
            cfg = from_dict(ModelConfig, params, Config(strict=True))
            enc = BaseGenerator.get_model(name, self.data_conf, cfg)
            hist_dim = getattr(getattr(enc, "encoder", None), "output_dim", None)
        else:
            raise ValueError("Not implemented any methods yet. Only GRU.")

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            msg = enc.load_state_dict(ckpt["model"], strict=True)
            logger.info("History encoder: " + str(msg))
            enc = freeze_module(enc)

        return enc, hist_dim

    def _select_indices(self, names_all: List[str]) -> List[int]:
        if not names_all:
            return []
        if not self.focus:
            return list(range(len(names_all)))
        return [i for i, name in enumerate(names_all) if name in self.focus]

    def _get_feature_layout(self, batch: GenBatch) -> FeatureLayout:
        num_names_all = list(batch.num_features_names or [])
        cat_names_all = list(batch.cat_features_names or [])

        num_idxs = self._select_indices(num_names_all)
        cat_idxs = self._select_indices(cat_names_all)

        num_names = []
        if (not self.focus) or (self.data_conf.time_name in self.focus):
            num_names.append(self.data_conf.time_name)
        num_names.extend(num_names_all[i] for i in num_idxs)
        cat_names = [cat_names_all[i] for i in cat_idxs]

        return FeatureLayout(
            num_names=num_names,
            cat_names=cat_names,
            num_idxs=num_idxs,
            cat_idxs=cat_idxs,
        )

    def feature_preprocess(self, hist: GenBatch, tgt: Optional[GenBatch] = None):
        layout = self._get_feature_layout(hist)
        if not layout.cat_idxs:
            raise ValueError(
                "At least one categorical feature is required for this model."
            )

        hist_num_features = []
        if self.data_conf.time_name in layout.num_names:
            hist_num_features.append(hist.time.unsqueeze(-1))
        if layout.num_idxs:
            if hist.num_features is None:
                missing = [hist.num_features_names[i] for i in layout.num_idxs]
                raise ValueError(
                    f"num_features tensor is None but layout expects: {missing}"
                )
            hist_num_features.append(hist.num_features[..., layout.num_idxs])
        if not hist_num_features:
            raise ValueError("At least one numerical feature is required for this model.")
        hist_num = torch.cat(hist_num_features, dim=-1)

        hist_cat = hist.cat_features[..., layout.cat_idxs]

        tgt_num = None
        tgt_cat = None
        if tgt is not None:
            tgt_num_features = []
            if self.data_conf.time_name in layout.num_names:
                tgt_num_features.append(tgt.time.unsqueeze(-1))
            if layout.num_idxs:
                if tgt.num_features is None:
                    missing = [tgt.num_features_names[i] for i in layout.num_idxs]
                    raise ValueError(
                        f"num_features tensor is None but layout expects: {missing}"
                    )
                tgt_num_features.append(tgt.num_features[..., layout.num_idxs])
            if tgt_num_features:
                tgt_num = torch.cat(tgt_num_features, dim=-1)
            tgt_cat = tgt.cat_features[..., layout.cat_idxs]

        hist_num = self._to_blf(hist_num)
        hist_cat = self._to_blf(hist_cat)
        if tgt_num is not None:
            tgt_num = self._to_blf(tgt_num)
        if tgt_cat is not None:
            tgt_cat = self._to_blf(tgt_cat)

        return tgt_num, tgt_cat, hist_num, hist_cat, layout

    def forward(self, x: GenBatch) -> torch.Tensor:
        x.time = x.time.float()
        x.target_time = x.target_time.float()

        embeddings = None
        if self.history_encoder is not None:
            embeddings = self.history_encoder.get_embeddings(x)

        hist = x.tail(self.history_len)
        tgt = x.get_target_batch()

        tgt_num, tgt_cat, hist_num, hist_cat, layout = self.feature_preprocess(hist, tgt)
        self.set_numerical_diff_mask(layout.num_names)
        loss = self.model.compute_loss(
            tgt_num, tgt_cat, hist_num, hist_cat, layout.cat_names, embeddings
        )
        return loss

    def set_numerical_diff_mask(self, num_order: List[str]):
        if not num_order:
            raise ValueError("Numerical feature order must not be empty.")

        if self.diff_features:
            diff_idx = [i for i, name in enumerate(num_order) if name in self.diff_features]
        else:
            diff_idx = list(range(len(num_order)))

        self.model.time_diff_.set_diffusion_mask(diff_idx, len(num_order))

    @torch.no_grad()
    def _generate_chunk(self, x: GenBatch) -> GenBatch:
        hist = x.tail(self.history_len)
        tgt = x.get_target_batch()

        embeddings = None
        if self.history_encoder is not None:
            embeddings = self.history_encoder.get_embeddings(x)

        tgt_num, tgt_cat, hist_num, hist_cat, layout = self.feature_preprocess(hist, tgt)
        self.set_numerical_diff_mask(layout.num_names)

        if tgt_num is not None and tgt_num.shape[1] != self.generation_len:
            tgt_num = None
            tgt_cat = None

        if self.help_net is not None:
            pred = self.help_net.generate(x, self.generation_len, topk=-1)
            provided_num_features = []
            if self.data_conf.time_name in layout.num_names:
                provided_num_features.append(pred.time.unsqueeze(-1))
            for num_feature in layout.num_names:
                if num_feature == self.data_conf.time_name:
                    continue
                if pred.num_features is None or pred.num_features_names is None:
                    raise ValueError(
                        f"help_net prediction missing num_features for {num_feature!r}"
                    )
                idx = pred.num_features_names.index(num_feature)
                provided_num_features.append(pred.num_features[..., [idx]])

            provided_cat_features = []
            for cat_feature in layout.cat_names:
                idx = pred.cat_features_names.index(cat_feature)
                provided_cat_features.append(pred.cat_features[..., [idx]])

            tgt_num = self._to_blf(torch.cat(provided_num_features, dim=-1).float())
            tgt_cat = self._to_blf(torch.cat(provided_cat_features, dim=-1).long())

        B, _, F_cat = hist_cat.shape
        _, _, F_num = hist_num.shape

        chunk_len = self.generation_len
        provide_cats = bool(set(layout.cat_names) & self.fix_features)
        with torch.cuda.amp.autocast():
            pred_cat, pred_num = self.model.sample(
                hist_num.repeat_interleave(self.repeat_samples, dim=0),
                hist_cat.repeat_interleave(self.repeat_samples, dim=0),
                chunk_len,
                layout.cat_names,
                tgt_e=tgt_cat.repeat_interleave(self.repeat_samples, dim=0) if provide_cats else None,
                tgt_x=tgt_num.repeat_interleave(self.repeat_samples, dim=0) if tgt_num is not None else None,
                hist_emb=embeddings.repeat_interleave(self.repeat_samples, dim=0) if embeddings is not None else None,
                cfg_w=self.cfg_w,
            )
        pred_cat = pred_cat.reshape(B, self.repeat_samples, chunk_len, F_cat).permute(0, 2, 3, 1)
        pred_num = pred_num.reshape(B, self.repeat_samples, chunk_len, F_num).permute(0, 2, 3, 1)

        cat = x.target_cat_features.clone()[:self.generation_len]
        num = (
            x.target_num_features.clone()[: self.generation_len]
            if x.target_num_features is not None
            else None
        )
        time = x.target_time.clone()[:self.generation_len]

        cat_local_idx = {name: i for i, name in enumerate(layout.cat_names)}
        for feature in layout.cat_names:
            if feature not in self.diff_features:
                continue
            global_idx = x.cat_features_names.index(feature)
            local_idx = cat_local_idx[feature]
            new_vals = pred_cat[..., local_idx, 0]
            cat[..., global_idx] = new_vals.long().permute(1, 0)

        num_local_idx = {name: i for i, name in enumerate(layout.num_names)}
        if self.data_conf.time_name in self.diff_features and self.data_conf.time_name in num_local_idx:
            time_idx = num_local_idx[self.data_conf.time_name]
            time = pred_num[..., time_idx, :].mean(dim=-1).permute(1, 0)

        for feature in layout.num_names:
            if feature == self.data_conf.time_name or feature not in self.diff_features:
                continue
            if num is None or x.num_features_names is None:
                raise ValueError(
                    f"Cannot write diffused numerical feature {feature!r}: "
                    "num_features is None"
                )
            local_idx = num_local_idx[feature]
            global_idx = x.num_features_names.index(feature)
            new_vals = pred_num[..., local_idx, 0]
            num[..., global_idx] = new_vals.float().permute(1, 0)

        sampled_batch = self.toGenBatch(
            cat, num, time, x.num_features_names, x.cat_features_names
        )
        return sampled_batch

    @torch.no_grad()
    def generate(
        self, x: GenBatch, gen_len: int, with_hist=False, **kwargs
    ) -> GenBatch:
        assert gen_len % self.generation_len == 0
        orig_hist = deepcopy(x) if with_hist else None
        x = deepcopy(x)

        total_gen = 0
        while total_gen < gen_len:
            sampled_batch = self._generate_chunk(x)
            x.append(sampled_batch)
            total_gen += self.generation_len

        pred = x.tail(total_gen).head(gen_len)
        if with_hist:
            orig_hist.append(pred)
            return orig_hist
        return pred

    def toGenBatch(self, cat, num, time, num_features_names, cat_features_names):
        s_length = torch.ones(cat.size(1)) * cat.size(0)
        assert time.ndim == 2
        return GenBatch(
            lengths=s_length,
            time=time,
            index=None,
            num_features=num,
            cat_features=cat,
            cat_features_names=cat_features_names,
            num_features_names=num_features_names,
        )
