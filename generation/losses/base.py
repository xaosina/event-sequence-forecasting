import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_linear_assignment import batch_linear_assignment

from generation.data.data_types import GenBatch, LatentDataConfig, PredBatch

from .utils import mse_valid, r1_valid, rse_valid, mae_valid


class BaseLoss(Module):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        num_weight: float = 0.5,
        ignore_index: int = -100,
        num_loss: str = "r1",
    ):
        super().__init__()
        assert 0 <= num_weight <= 1
        self.data_conf = data_conf
        self.num_weight = num_weight
        self._ignore_index = ignore_index
        if num_loss == "r1":
            self.num_loss = r1_valid
        elif num_loss == "r2":
            self.num_loss = rse_valid
        elif num_loss == "mse":
            self.num_loss = mse_valid
        elif num_loss == "mae":
            self.num_loss = mae_valid
        else:
            raise NotImplementedError

    @property
    def pred_slice(self):
        """Slice applied to prediction tensors (e.g., slice(None, -1))"""
        return slice(None)  # Default: no slicing

    @property
    def true_slice(self):
        """Slice applied to target tensors (e.g., slice(1, None))"""
        return slice(None)  # Default: no slicing

    def _compute_mse(
        self, y_true: GenBatch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mse_sum = 0.0
        mse_count = 0

        data_conf = self.data_conf
        num_names = y_pred.num_features_names or []
        num_names = list(set(data_conf.focus_on) & set(num_names))
        if data_conf.time_name in data_conf.focus_on:
            pred_time = y_pred.time[self.pred_slice]  # [L, B]
            true_time = y_true.time[self.true_slice]  # [L, B]
            current_mask = valid_mask[self.true_slice]
            loss, count = self.num_loss(pred_time, true_time, current_mask)
            mse_sum += loss
            mse_count += count

        if num_names:
            true_feature_ids = [
                y_true.num_features_names.index(name) for name in num_names
            ]
            pred_feature_ids = [
                y_pred.num_features_names.index(name) for name in num_names
            ]
            pred_num = y_pred.num_features[self.pred_slice, :, pred_feature_ids]
            true_num = y_true.num_features[self.true_slice, :, true_feature_ids]
            current_mask = valid_mask[self.true_slice]
            loss, count = self.num_loss(pred_num, true_num, current_mask)
            mse_sum += loss
            mse_count += count

        return (mse_sum / mse_count) if mse_count != 0 else torch.tensor(0.0)

    def _compute_ce(
        self, y_true: GenBatch, y_pred: PredBatch, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=valid_mask.device)
        data_conf = self.data_conf
        cat_names = y_pred.cat_features or {}
        cat_names = list(set(data_conf.focus_on) & set(cat_names))

        total_ce = 0.0
        ce_count = 0
        for key in cat_names:
            true_cat = y_true[key][self.true_slice].clone()
            current_mask = valid_mask[self.true_slice]
            true_cat[~current_mask] = self._ignore_index

            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            pred_cat = pred_cat[:, :, self.pred_slice]
            # Compute loss
            ce_loss = F.cross_entropy(
                pred_cat,
                true_cat.permute(1, 0),  # [B, L']
                ignore_index=self._ignore_index,
            )
            total_ce += ce_loss
            ce_count += 1

        return (total_ce / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def _valid_mask(self, y_true) -> torch.Tensor:
        return (
            torch.arange(y_true.lengths.max(), device=y_true.lengths.device)[:, None]
            < y_true.lengths
        )

    def __call__(self, y_true, y_pred) -> dict:
        valid_mask = self._valid_mask(y_true)
        mse_loss = self._compute_mse(y_true, y_pred, valid_mask)
        ce_loss = self._compute_ce(y_true, y_pred, valid_mask)
        return {"loss": self.combine_losses(mse_loss, ce_loss)}

    def combine_losses(self, mse_loss, ce_loss):
        return 2 * (self.num_weight * mse_loss + (1 - self.num_weight) * ce_loss)


class BaselineLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(None, -1)  # Predictions use all except last time step

    @property
    def true_slice(self):
        return slice(1, None)  # Targets use all except first time step


class NoOrderLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(None, -1)  # Predictions use all except last time step

    @property
    def true_slice(self):
        return slice(1, None)  # Targets use all except first time step

    def bag_mse_loss(self, logits, targets, window=10, ignore_index=-100, eps=1e-8):
        """
        logits  : [B,L,C]
        targets : [B,L]
        Возвращает средний MSE между нормализованными гистограммами p и q.
        """
        B, L, C = logits.shape
        prob = logits.softmax(-1)  # [B,L,C]

        loss, n_windows = 0.0, 0
        for t0 in range(0, L, window):
            t1 = min(L, t0 + window)

            # Распределение предсказанний
            p_counts = prob[:, t0:t1].sum(1)  # [B,C]
            p = p_counts / (p_counts.sum(-1, keepdim=True) + eps)

            # Распределение таргеров
            tgt = targets[:, t0:t1]
            mask = tgt.ne(ignore_index)
            one_hot = F.one_hot(tgt.clamp(min=0), C).float() * mask.unsqueeze(-1)
            q_counts = one_hot.sum(1)  # [B,C]
            q = q_counts / (q_counts.sum(-1, keepdim=True) + eps)

            loss += ((p - q) ** 2).sum(-1).mean() / 2  # MSE
            n_windows += 1

        return loss / max(n_windows, 1)

    def _compute_loss(self, y_true, y_pred, valid_mask):
        if not y_pred.cat_features:
            return torch.tensor(0.0, device=valid_mask.device)
        data_conf = self.data_conf
        cat_names = y_pred.cat_features or {}
        cat_names = list(set(data_conf.focus_on) & set(cat_names))

        total_loss = 0.0
        ce_count = 0
        assert len(cat_names) == 1

        for key in cat_names:
            true_cat = y_true[key][self.true_slice].clone()
            current_mask = valid_mask[self.true_slice]
            true_cat[~current_mask] = self._ignore_index

            pred_cat = y_pred.cat_features[key].permute(1, 2, 0)  # [B, C, L]
            pred_cat = pred_cat[:, :, self.pred_slice]

            loss = self.bag_mse_loss(
                pred_cat.permute(0, 2, 1), true_cat.permute(1, 0), window=int(1e10)
            )
            total_loss += loss
            ce_count += 1

        return (total_loss / ce_count) if ce_count != 0 else torch.tensor(0.0)

    def __call__(self, y_true, y_pred) -> torch.Tensor:
        valid_mask = self._valid_mask(y_true)
        loss = self._compute_loss(y_true, y_pred, valid_mask)
        return {"loss": loss}


class TailLoss(BaseLoss):
    @property
    def pred_slice(self):
        return slice(-1 - self.data_conf.generation_len, -1)  # Take shifted last

    @property
    def true_slice(self):
        return slice(-self.data_conf.generation_len, None)  # Predict only generated_len


class VAELoss(BaseLoss):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        num_weight: float = 0.5,
        init_beta: float = 1.0,
        ignore_index: int = -100,
        num_loss: str = "r1",
    ):
        super().__init__(data_conf, num_weight, ignore_index, num_loss)
        self._beta = init_beta

    def __call__(self, y_true: GenBatch, data) -> torch.Tensor:
        y_pred, params = data
        base_loss = super().__call__(y_true, y_pred)["loss"]

        mu_z = params["mu_z"]
        std_z = params["std_z"]
        # TODO:
        # correct version of computing kld term:
        # kld_term = -0.5 * (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(0).sum()
        # legacy version of computing kld term:
        kld_term = -0.5 * torch.mean(
            (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(-1).mean()
        )

        return {"loss": base_loss + self._beta * kld_term, "kl_loss": kld_term}

    def update_beta(self, value):
        self._beta = value


class SeqVAELoss(VAELoss):
    def __call__(self, y_true: GenBatch, data) -> torch.Tensor:
        y_true = y_true.tail(self.data_conf.generation_len)
        return super().__call__(y_true, data)

class SeqVAEOTDLoss(VAELoss):

    def __call__(self, y_true: GenBatch, data):
        y_true = y_true.tail(self.data_conf.generation_len)
        y_pred, params = data
        base_loss = self._matches_loss(y_true, y_pred)

        mu_z = params["mu_z"]
        std_z = params["std_z"]
        # TODO:
        # correct version of computing kld term:
        # kld_term = -0.5 * (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(0).sum()
        # legacy version of computing kld term:
        kld_term = -0.5 * torch.mean(
            (1 + std_z - mu_z.pow(2) - std_z.exp()).mean(-1).mean()
        )

        return {"loss": base_loss + self._beta * kld_term, "kl_loss": kld_term}
    
    def _matches_loss(self, y_true, y_pred):
        T, B = y_true.shape
        K, _ = y_pred.shape
        assert K == T
        losses = {}
        # 1. Compute cat
        for name in self.data_conf.focus_cat:
            pred = y_pred.cat_features[name].movedim(0, -1)  # B, C, K
            true = y_true[name].T[:, None]  # B, 1, T
            C = pred.shape[1]
            pred = pred.unsqueeze(-1).expand(B, C, K, T)
            true = true.expand(B, K, T)
            ce_loss = F.cross_entropy(pred, true, reduction="none")
            losses[name] = ce_loss

        # 2. Compute num
        for name in self.data_conf.focus_num:
            if name == self.data_conf.time_name:
                pred = y_pred.time.T  # B, K
                true = y_true.time.T  # B, T
                # true = true[:, 1:] - true[:, :1]  # delta from last seen
            else:
                pred = y_pred[name].T  # B, K
                true = y_true[name].T  # B, T
            pred, true = pred.reshape(B, K, 1), true.reshape(B, 1, T)
            losses[name] = (pred - true).abs()

        cost = torch.zeros((B, K, T), device=y_true.device)
        for name in losses:
            cost += losses[name]

        if cost.isnan().any():
            return torch.tensor(float("nan"))

        assignment = batch_linear_assignment(cost).unsqueeze(-1)  # B, K, 1

        losses = list(losses.values())
        loss = sum(losses[1:], start=losses[0])
        loss = loss.take_along_dim(assignment.clip(min=0), -1)  # V, K, 1
        loss = loss[assignment >= 0].sum()  # V * T
        final_loss = loss / B
        return final_loss

class AELoss(BaseLoss):
    def __init__(
        self,
        data_conf: LatentDataConfig,
        num_weight: float = 0.5,
        l2_coef: float = 0.001,
        ignore_index: int = -100,
        num_loss: str = "r1",
    ):
        super().__init__(data_conf, num_weight, ignore_index, num_loss)
        self.l2_coef = l2_coef
        print(l2_coef)

    def __call__(self, y_true: GenBatch, data) -> torch.Tensor:
        y_pred, hidden = data
        base_loss = super().__call__(y_true, y_pred)["loss"]

        hidden = hidden[y_true.valid_mask]  # L*B, D
        l2_term = (hidden**2).sum(-1).mean(0)

        return {"loss": base_loss + self.l2_coef * l2_term, "l2_term": l2_term}
