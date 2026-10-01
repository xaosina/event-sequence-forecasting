import math
from copy import deepcopy
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from ebes.types import Seq

from generation.data import batch_tfs
from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch
from generation.data.utils import create_instances_from_module
from generation.models.autoencoders.base import AEConfig, BaseAE

from .modules import Tokenizer, Transformer, Reconstructor
from .utils import get_features_after_transform


class SeqVAE(BaseAE):
    def __init__(self, data_conf: LatentDataConfig, model_config):
        super().__init__()
        self.gen_len = data_conf.generation_len
        model_config: AEConfig = model_config.autoencoder
        vae_params = {
            "num_layers": model_config.params.get("num_layers", 2),
            "d_token": model_config.params.get("d_token", 6),
            "n_head": model_config.params.get("n_head", 2),
            "factor": model_config.params.get("factor", 32),
            "gen_len": self.gen_len,
        }
        batch_transforms = create_instances_from_module(
            batch_tfs, model_config.batch_transforms
        )
        num_names, cat_cardinalities = get_features_after_transform(
            data_conf, batch_transforms, model_config
        )

        self.encoder = SeqEncoder(
            **vae_params,
            use_time=model_config.params["use_time"],
            pretrain=model_config.pretrain,
            frozen=model_config.frozen,
            cat_cardinalities=cat_cardinalities,
            num_features=num_names,
            batch_transforms=batch_transforms,
            always_reparametrize=model_config.params.get("always_reparametrize", False),
        )
        self.model_config = model_config

        self.decoder = SeqDecoder(
            **vae_params,
            cat_cardinalities=cat_cardinalities,
            num_names=num_names,
            batch_transforms=batch_transforms,
            linear_num=model_config.params.get("linear_num", False)
        )

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Variational AutoEncoder
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """

        assert not self.encoder.pretrained
        orig_hist = x.anti_tail(self.gen_len)
        x, params = self.encoder(x, reparametrize=True)
        x = self.decoder(x, orig_hist)
        return x, params

    def generate(
        self,
        hist: GenBatch,
        gen_len: int,
        with_hist=False,
        topk=1,
        temperature=1.0,
    ) -> GenBatch:
        batch = deepcopy(hist)
        assert hist.target_time.shape[0] == gen_len, hist.target_time.shape
        batch.append(batch.get_target_batch())
        x = self.encoder(batch)
        # if not self.encoder.pretrained:
        #     x = x[0]
        x = self.decoder.generate(x, hist, topk=topk, temperature=temperature)
        if with_hist:
            raise NotImplementedError
            hist.append(x)
            return hist
        else:
            return x


class SeqEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_token: int = 6,
        n_head: int = 2,
        factor: int = 32,
        gen_len: int = 32,
        use_time: bool = True,
        pretrain: bool = False,
        frozen: bool = False,
        cat_cardinalities: Mapping[str, int] | None = None,
        num_features: Sequence[str] | None = None,
        batch_transforms: list | None = None,
        bias=True,
        always_reparametrize=False,
    ):
        super(SeqEncoder, self).__init__()
        self.d_token = d_token
        self.pretrained = not pretrain
        self.frozen = frozen
        self.gen_len = gen_len

        if num_features is not None:
            num_count = len(num_features)
        else:
            num_count = 0

        self.use_time = use_time
        # if use_time:
        num_count += 1
        self.batch_transforms = batch_transforms

        self.tokenizer = Tokenizer(
            d_numerical=num_count,
            categories=list(cat_cardinalities.values()) if cat_cardinalities else None,
            d_token=d_token,
            bias=bias,
        )
        self._out_dim = (num_count + len(cat_cardinalities)) * self.d_token

        self.encoder_mu = Transformer(
            num_layers, self._out_dim, n_head, d_token, factor
        )
        self.encoder_std = None
        if not self.pretrained:
            self.encoder_std = Transformer(
                num_layers, self._out_dim, n_head, d_token, factor
            )
        self.always_reparametrize = always_reparametrize

    @property
    def output_dim(self):
        return self._out_dim

    def reparametrize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: GenBatch, copy=True, reparametrize=False) -> Seq:
        if copy:
            batch = deepcopy(batch)

        if self.batch_transforms:
            for tf in self.batch_transforms:
                tf(batch)

        pos = batch.lengths - self.gen_len - 1
        last_time = torch.where(
            pos >= 0,
            batch.time[pos, torch.arange(batch.time.size(1), device=batch.device)],
            torch.zeros_like(pos, dtype=batch.time.dtype),
        )
        batch = batch.tail(self.gen_len)
        batch.time = batch.time - last_time[None]

        x_num, x_cat = batch.num_features, batch.cat_features

        x_num = [] if x_num is None else [x_num]
        if self.use_time:
            time = batch.time
        else:
            time = torch.zeros_like(batch.time)
        x_num = [time.unsqueeze(-1)] + x_num  # L, B, 1
        x_num = torch.cat(x_num, dim=2)

        D_num = x_num.size(-1) if x_num is not None else 0
        D_cat = x_cat.size(-1) if x_cat is not None else 0
        D_all, d_token = D_num + D_cat, self.d_token
        time = batch.time
        L, B = time.shape
        D_vae = (D_num + D_cat) * d_token
        assert L > 0, "No features for VAE training"
        valid_mask = batch.valid_mask.ravel()  # L * B

        if x_num is not None:
            x_num = x_num.transpose(0, 1).reshape(-1, D_num)[valid_mask]  # B*L, D_num
        if x_cat is not None:
            x_cat = x_cat.transpose(0, 1).reshape(-1, D_cat)[valid_mask]  # B*L, D_cat

        x = self.tokenizer(x_num, x_cat)  # (B*L), D_num + D_cat, d_token
        x = x.view(B, L, D_all, d_token).view(
            B, L, D_all * d_token
        )  # B, L, D * d_token
        mu_z = self.encoder_mu(x)  # B, L, D * d_token

        # Handle VAE output based on training mode
        if not (reparametrize or self.always_reparametrize):
            output = mu_z
        else:
            std_z = self.encoder_std(x).clamp(max=70)
            output = self.reparametrize(mu_z, std_z)

        output = output.transpose(0, 1)
        # Prepare return values
        seq = Seq(tokens=output, lengths=batch.lengths, time=time)
        if not reparametrize:
            return seq
        return (seq, {"mu_z": mu_z, "std_z": std_z})


class SeqDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_token: int = 6,
        n_head: int = 2,
        factor: int = 32,
        gen_len: int = 32,
        num_names: Sequence[str] | None = None,
        cat_cardinalities: Mapping[str, int] = None,
        batch_transforms: list | None = None,
        linear_num: bool = False,
    ):
        super(SeqDecoder, self).__init__()
        self.batch_transforms = batch_transforms
        self.d_token = d_token
        self.gen_len = gen_len
        self.num_names = num_names
        self.cat_cardinalities = cat_cardinalities
        num_counts = 1  # Time
        if self.num_names:
            num_counts += len(self.num_names)
        token_dim = (num_counts + len(cat_cardinalities)) * self.d_token
        self.decoder = Transformer(num_layers, token_dim, n_head, d_token, factor)
        self.reconstructor = Reconstructor(
            num_counts,
            self.cat_cardinalities or {},
            d_token=d_token,
            linear_num=linear_num,
        )

    def forward(self, seq: Seq, hist: GenBatch) -> PredBatch:
        x = seq.tokens
        L, B, D_vae = x.shape
        d_token = self.d_token
        D = D_vae // d_token
        assert D * d_token == D_vae, "Invalid token dimensions"
        # Process sequence mask and reshape inputs
        assert seq.lengths.min() == L
        x = x.transpose(0, 1)

        # Decode and reconstruct
        h = self.decoder(x)  # B, L, D * d_token
        h = h.view(B, L, D, d_token).transpose(0, 1).reshape(L * B, D, d_token)
        recon_num, recon_cat = self.reconstructor(h)

        # Prepare numerical features
        num_features = None
        recon_num = recon_num.view(L, B, -1)

        time = recon_num[:, :, 0]
        pos = hist.lengths - 1
        last_time = torch.where(
            pos >= 0,
            hist.time[pos, torch.arange(hist.time.size(1), device=hist.device)],
            torch.zeros_like(pos, dtype=hist.time.dtype),
        )
        time += last_time[None]

        if self.num_names:
            D_num = len(self.num_names)
            num_features = recon_num[:, :, 1 : D_num + 1]

        # Prepare categorical features
        cat_features = {}
        if self.cat_cardinalities:
            for name, cat in recon_cat.items():
                D_cat = cat.shape[-1]
                cat_features[name] = cat.view(L, B, D_cat)
        return PredBatch(
            lengths=seq.lengths,
            time=time,
            num_features=num_features,
            num_features_names=self.num_names,
            cat_features=cat_features if cat_features else None,
        )

    def generate(self, seq: Seq, orig_hist: GenBatch, topk=1, temperature=1.0) -> GenBatch:
        batch = self.forward(seq, hist=orig_hist).to_batch(topk, temperature)
        if self.batch_transforms is not None:
            if orig_hist is not None:
                batch_len = batch.shape[0]
                hist = deepcopy(orig_hist)
                for tf in self.batch_transforms:
                    tf(hist)
                hist.append(batch)
                batch = hist
            for tf in reversed(self.batch_transforms):
                tf.reverse(batch)
            if hist is not None:
                batch = batch.tail(batch_len)
        return batch
