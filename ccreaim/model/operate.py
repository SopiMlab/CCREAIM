from typing import Union

import torch
from torch.nn import functional as F

from ..utils import util
from ..utils.cfg_classes import HyperConfig
from . import ae, e2e, e2e_chunked, transformer, vae, vqvae


def step(
    model: torch.nn.Module,
    batch: Union[tuple[torch.Tensor, str], tuple[torch.Tensor, str, torch.Tensor]],
    device: torch.device,
    hyper_cfg: HyperConfig,
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
        spec_weight = hyper_cfg.spectral_loss.weight
        multi_spec = util.multispectral_loss(tgt, pred, hyper_cfg.spectral_loss)
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
        kld_weight = hyper_cfg.kld_loss.weight
        kld = -0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        spec_weight = hyper_cfg.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_kld": float((kld_weight * kld).item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + kld_weight * kld + spec_weight * multi_spec
    elif isinstance(model, vqvae.VQVAE):
        seq, _ = batch
        seq = seq.to(device)
        pred, latents, quantized_latents = model(seq)
        mse = F.mse_loss(pred, seq)
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * hyper_cfg.vqvae.beta + embedding_loss

        # Add the residue back to the latents

        spec_weight = hyper_cfg.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "commitment_loss": commitment_loss.item(),
                "embedding_loss": hyper_cfg.vqvae.beta * embedding_loss.item(),
                "vq_loss": vq_loss.item(),
                "loss_mse": float(mse.item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + spec_weight * multi_spec + vq_loss
    else:
        seq, _ = batch
        seq = seq.to(device)
        pred = model(seq)
        mse = F.mse_loss(pred, seq)
        spec_weight = hyper_cfg.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + spec_weight * multi_spec
    return loss, pred, info


def get_model_init_function(hyper_cfg: HyperConfig):
    # Model init function mapping
    if hyper_cfg.model == "ae":
        get_model = lambda: ae.get_autoencoder("base", hyper_cfg)
    elif hyper_cfg.model == "res-ae":
        get_model = lambda: ae.get_autoencoder("res-ae", hyper_cfg)
    elif hyper_cfg.model == "vae":
        get_model = lambda: vae.get_vae("base", hyper_cfg)
    elif hyper_cfg.model == "res-vae":
        get_model = lambda: vae.get_vae("res-vae", hyper_cfg)
    elif hyper_cfg.model == "vq-vae":
        get_model = lambda: vqvae.get_vqvae("base", hyper_cfg)
    elif hyper_cfg.model == "transformer":
        get_model = lambda: transformer.get_transformer("base", hyper_cfg)
    elif hyper_cfg.model == "e2e":
        get_model = lambda: e2e.get_e2e("base_ae", hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked":
        get_model = lambda: e2e_chunked.get_e2e_chunked("base_ae", hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-ae":
        get_model = lambda: e2e_chunked.get_e2e_chunked("base_res-ae", hyper_cfg)
    else:
        raise ValueError(f"Model type {hyper_cfg.model} is not defined!")
    return get_model
