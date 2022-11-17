from typing import Union

import torch
from torch.nn import functional as F

from ..model import ae, e2e_chunked, transformer, vae, vqvae
from ..utils import cfg_classes, util


def step(
    model: torch.nn.Module,
    batch: Union[tuple[torch.Tensor, str], tuple[torch.Tensor, str, torch.Tensor]],
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}
    if isinstance(model, transformer.Transformer):
        seq, _ = batch
        seq = seq.to(device)
        src = seq[:, :-1, :]
        tgt = seq[:, 1:, :]
        tgt_mask = model.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        pred = model(src, tgt, tgt_mask)
        loss = F.mse_loss(pred, tgt)
    elif isinstance(model, e2e_chunked.E2EChunked):
        seq, _, pad_mask = batch
        seq = seq.to(device)
        pad_mask = pad_mask.to(device)
        pred = model(seq, pad_mask)
        tgt = seq[:, 1:, :]
        tgt_pad_mask = pad_mask[:, 1:]
        mse = F.mse_loss(pred, tgt, reduction="none")
        mse[tgt_pad_mask] = 0
        mse = mse.mean()
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = util.multispectral_loss(tgt, pred, cfg)
        multi_spec[tgt_pad_mask] = 0
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_spectral": spec_weight * multi_spec.item(),
            }
        )
        loss = mse + spec_weight * multi_spec
    elif isinstance(model, vae.VAE):
        seq, _ = batch
        seq = seq.to(device)
        pred, mu, sigma = model(seq)
        mse = F.mse_loss(pred, seq)
        kld_weight = cfg.hyper.kld_loss.weight
        kld = -0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, cfg)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_kld": float((kld_weight * kld).item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + kld_weight * kld + spec_weight * multi_spec
    else:
        seq, _ = batch
        seq = seq.to(device)
        pred = model(seq)
        mse = F.mse_loss(pred, seq)
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, cfg)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + spec_weight * multi_spec
    return loss, pred, info
