import logging
from typing import Union

import torch
from torch.nn import functional as F

from ..utils import util
from ..utils.cfg_classes import HyperConfig
from . import ae, e2e, e2e_chunked, transformer, vae, vqvae

log = logging.getLogger(__name__)


def step(
    model: torch.nn.Module,
    batch: Union[tuple[torch.Tensor, str], tuple[torch.Tensor, str, torch.Tensor]],
    device: torch.device,
    hyper_cfg: HyperConfig,
    batchnum: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}
    if isinstance(model, transformer.Transformer):
        seq, _, inds = batch
        seq = seq.to(device)
        inds = inds.to(device)
        src = seq
        tgt = torch.cat(
            (
                torch.zeros_like(seq[:, 0:1, :], device=seq.device),
                seq[:, :-1, :],
            ),
            dim=1,
        )
        tgt_mask = model.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        pred = model(src, tgt, tgt_mask=tgt_mask)

        if hyper_cfg.transformer.linear_map:
            pred = pred.view(-1, hyper_cfg.vqvae.num_embeddings)
            inds = inds.view(-1)
            trf_auto = F.cross_entropy(pred, inds)
        else:
            trf_auto = F.mse_loss(pred, src)

        if batchnum % 2000 == 0:
            vq_inds = inds
            log.info(f"vq_inds: {vq_inds}")
            trf_inds_prob = pred
            trf_inds = trf_inds_prob.argmax(-1)
            log.info(f"trf_inds: {trf_inds}")
            vq_inds_counts = dict()
            for i in torch.flatten(vq_inds):
                vq_inds_counts[i.item()] = vq_inds_counts.get(i.item(), 0) + 1
            trf_inds_counts = dict()
            for i in torch.flatten(trf_inds):
                trf_inds_counts[i.item()] = trf_inds_counts.get(i.item(), 0) + 1

            vq_sorted = {
                k: v
                for k, v in sorted(
                    vq_inds_counts.items(), key=lambda item: item[1], reverse=True
                )
            }
            trf_sorted = {
                k: v
                for k, v in sorted(
                    trf_inds_counts.items(), key=lambda item: item[1], reverse=True
                )
            }
            log.info(f"Counts of specific indices in vq_inds: {vq_sorted}")
            log.info(f"Counts of specific indices in trf_inds: {trf_sorted}")

        loss = trf_auto

    elif isinstance(model, e2e_chunked.E2EChunked):
        seq, _, pad_mask = batch
        seq = seq.to(device)
        pad_mask = pad_mask.to(device)
        pred, enc_out, trf_out = model(seq, pad_mask)
        mse = F.mse_loss(pred, seq, reduction="none")
        mse[pad_mask] = 0
        mse = mse.mean()
        trf_auto_mse = torch.tensor(0)
        if hyper_cfg.transformer.autoregressive_loss_weight:
            trf_auto_mse = F.mse_loss(enc_out, trf_out, reduction="none")
            trf_auto_mse[pad_mask] = 0
            trf_auto_mse = trf_auto_mse.mean()
            info.update(
                {
                    "train/loss_transformer_auto_mse": hyper_cfg.transformer.autoregressive_loss_weight
                    * float(trf_auto_mse.item())
                }
            )
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec[pad_mask] = 0
        multi_spec = multi_spec.mean()
        info.update(
            {
                "train/loss_mse": float(mse.item()),
                "train/loss_multi_spectral": hyper_cfg.spectral_loss.weight
                * multi_spec.item(),
            }
        )

        loss = (
            mse
            + hyper_cfg.spectral_loss.weight * multi_spec
            + hyper_cfg.transformer.autoregressive_loss_weight * trf_auto_mse
        )

    elif isinstance(model, e2e_chunked.E2EChunkedVQVAE):
        seq, _, pad_mask = batch
        seq = seq.to(device)
        pad_mask = pad_mask.to(device)
        pred, enc_out, vq_out, vq_inds, trf_inds_prob = model(seq, pad_mask)
        # Compute the VQ Losses
        if not hyper_cfg.freeze_pre_trained:
            commitment_loss = F.mse_loss(vq_out.detach(), enc_out)
            embedding_loss = F.mse_loss(vq_out, enc_out.detach())
        else:
            commitment_loss = torch.tensor(0)
            embedding_loss = torch.tensor(0)
            model.vq.embedding.grad = 0

        vq_loss = hyper_cfg.vqvae.beta * commitment_loss + embedding_loss
        mse = F.mse_loss(pred, seq, reduction="none")
        mse[pad_mask] = 0
        mse = mse.mean()
        trf_auto_ce = torch.tensor(0)
        if hyper_cfg.transformer.autoregressive_loss_weight:
            trf_inds_prob = trf_inds_prob.view(-1, hyper_cfg.vqvae.num_embeddings)
            vq_inds = vq_inds.view(-1)
            trf_auto_ce = F.cross_entropy(trf_inds_prob, vq_inds, reduction="none")
            trf_auto_ce = trf_auto_ce.view(-1, hyper_cfg.num_seq, model.enc_out_length)
            trf_auto_ce[pad_mask] = 0
            trf_auto_ce = trf_auto_ce.mean()
            info.update(
                {
                    "train/loss_transformer_auto_ce": hyper_cfg.transformer.autoregressive_loss_weight
                    * float(trf_auto_ce.item())
                }
            )
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec[pad_mask] = 0
        multi_spec = multi_spec.mean()
        info.update(
            {
                "train/loss_mse": float(mse.item()),
                "train/loss_multi_spectral": hyper_cfg.spectral_loss.weight
                * multi_spec.item(),
                "train/commitment_loss": hyper_cfg.vqvae.beta * commitment_loss.item(),
                "train/embedding_loss": embedding_loss.item(),
            }
        )
        if batchnum % 2000 == 0:
            log.info(f"vq_inds: {vq_inds}")
            trf_inds = trf_inds_prob.argmax(-1)
            log.info(f"trf_inds: {trf_inds}")
            vq_inds_counts = dict()
            for i in torch.flatten(vq_inds):
                vq_inds_counts[i.item()] = vq_inds_counts.get(i.item(), 0) + 1
            trf_inds_counts = dict()
            for i in torch.flatten(trf_inds):
                trf_inds_counts[i.item()] = trf_inds_counts.get(i.item(), 0) + 1

            vq_sorted = {
                k: v
                for k, v in sorted(
                    vq_inds_counts.items(), key=lambda item: item[1], reverse=True
                )
            }
            trf_sorted = {
                k: v
                for k, v in sorted(
                    trf_inds_counts.items(), key=lambda item: item[1], reverse=True
                )
            }
            log.info(f"Counts of specific indices in vq_inds: {vq_sorted}")
            log.info(f"Counts of specific indices in trf_inds: {trf_sorted}")

        loss = (
            mse
            + hyper_cfg.spectral_loss.weight * multi_spec
            + vq_loss
            + hyper_cfg.transformer.autoregressive_loss_weight * trf_auto_ce
        )

    elif isinstance(model, vae.VAE):
        seq, _ = batch
        seq = seq.to(device)
        pred, mu, sigma = model(seq)
        mae = torch.abs(pred - seq).mean()
        mse = F.mse_loss(pred, seq)
        kld_weight = hyper_cfg.kld_loss.weight
        kld = -0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        spec_weight = hyper_cfg.spectral_loss.weight
        spec_conv = util.spectral_convergence(seq, pred, hyper_cfg.spectral_loss)
        spec_conv = spec_conv.mean()
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "train/loss_mae": float(mae.item()),
                "train/loss_mse": float(mse.item()),
                "train/loss_kld": float((kld_weight * kld).item()),
                "train/loss_spectral_convergence": float(
                    spec_weight * spec_conv.item()
                ),
                "train/loss_multi_spectral": float(spec_weight * multi_spec.item()),
            }
        )

        loss = (
            mse
            + mae
            + kld_weight * kld
            + spec_weight * spec_conv
            + spec_weight * multi_spec
        )

    elif isinstance(model, vqvae.VQVAE):
        seq, _ = batch
        seq = seq.to(device)
        pred, latents, quantized_latents = model(seq)
        mse = F.mse_loss(pred, seq)
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = hyper_cfg.vqvae.beta * commitment_loss + embedding_loss

        spec_weight = hyper_cfg.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "train/commitment_loss": hyper_cfg.vqvae.beta * commitment_loss.item(),
                "train/embedding_loss": embedding_loss.item(),
                "train/vq_loss": vq_loss.item(),
                "train/loss_mse": float(mse.item()),
                "train/loss_spectral": float(spec_weight * multi_spec.item()),
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
                "train/loss_mse": float(mse.item()),
                "train/loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )

        loss = mse + spec_weight * multi_spec

    return loss, pred, info


def get_model_init_function(hyper_cfg: HyperConfig):
    # Model init function mapping
    if hyper_cfg.model == "ae":
        get_model = lambda: ae.get_autoencoder(hyper_cfg)
    elif hyper_cfg.model == "res-ae":
        get_model = lambda: ae.get_autoencoder(hyper_cfg)
    elif hyper_cfg.model == "vae":
        get_model = lambda: vae.get_vae(hyper_cfg)
    elif hyper_cfg.model == "res-vae":
        get_model = lambda: vae.get_vae(hyper_cfg)
    elif hyper_cfg.model == "vq-vae":
        get_model = lambda: vqvae.get_vqvae(hyper_cfg)
    elif hyper_cfg.model == "res-vqvae":
        get_model = lambda: vqvae.get_vqvae(hyper_cfg)
    elif hyper_cfg.model == "transformer":
        get_model = lambda: transformer.get_transformer(hyper_cfg)
    elif hyper_cfg.model == "e2e":
        get_model = lambda: e2e.get_e2e(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked":
        get_model = lambda: e2e_chunked.get_e2e_chunked(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-ae":
        get_model = lambda: e2e_chunked.get_e2e_chunked(hyper_cfg)
    elif hyper_cfg.model == "e2e-chunked_res-vqvae":
        get_model = lambda: e2e_chunked.get_e2e_chunked(hyper_cfg)
    else:
        raise ValueError(f"Model type {hyper_cfg.model} is not defined!")
    return get_model
