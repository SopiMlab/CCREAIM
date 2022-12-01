from typing import Union

import torch
from torch.nn import functional as F

from ..utils import util
from ..utils.cfg_classes import HyperConfig
from ..utils.rave_core import get_beta_kl_cyclic_annealed
from . import ae, e2e, e2e_chunked, rave, transformer, vae, vqvae

NUM_STEPS: int = 0


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
                "train/loss_mse": float(mse.item()),
                "train/loss_spectral": spec_weight * multi_spec.item(),
            }
        )
        loss = mse + spec_weight * multi_spec
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

        vq_loss = commitment_loss * hyper_cfg.vqvae.beta + embedding_loss

        # Add the residue back to the latents

        spec_weight = hyper_cfg.spectral_loss.weight
        multi_spec = util.multispectral_loss(seq, pred, hyper_cfg.spectral_loss)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "train/commitment_loss": commitment_loss.item(),
                "train/embedding_loss": hyper_cfg.vqvae.beta * embedding_loss.item(),
                "train/vq_loss": vq_loss.item(),
                "train/loss_mse": float(mse.item()),
                "train/loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        loss = mse + spec_weight * multi_spec + vq_loss
    elif isinstance(model, rave.RAVE):
        x, _ = batch
        x = x.to(device)
        if model.pqmf is not None:  # MULTIBAND DECOMPOSITION
            x = model.pqmf(x)

        if model.warmed_up:  # EVAL ENCODER
            model.encoder.eval()
        enc = model.encoder(x)
        z, kl = model.reparametrize(*enc)

        if model.warmed_up:  # FREEZE ENCODER
            z = z.detach()
            kl = kl.detach()

        # DECODE LATENT
        pred = model.decoder(z, add_noise=model.warmed_up)

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distance = model.distance(x, pred)

        if model.pqmf is not None:  # FULL BAND RECOMPOSITION
            x = model.pqmf.inverse(x)
            pred = model.pqmf.inverse(pred)
            distance = distance + model.distance(x, pred)

        loud_x = model.loudness(x)
        loud_y = model.loudness(pred)
        loud_dist = (loud_x - loud_y).pow(2).mean()
        distance = distance + loud_dist

        feature_matching_distance = 0.0
        if False:  # model.warmed_up:  # DISCRIMINATION
            feature_true = model.discriminator(x)
            feature_fake = model.discriminator(y)

            loss_dis = 0
            loss_adv = 0

            pred_true = 0
            pred_fake = 0

            for scale_true, scale_fake in zip(feature_true, feature_fake):
                feature_matching_distance = feature_matching_distance + 10 * sum(
                    map(
                        lambda x, y: abs(x - y).mean(),
                        scale_true,
                        scale_fake,
                    )
                ) / len(scale_true)

                _dis, _adv = model.adversarial_combine(
                    scale_true[-1],
                    scale_fake[-1],
                    mode=model.mode,
                )

                pred_true = pred_true + scale_true[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

        else:
            pred_true = torch.tensor(0.0).to(x)
            pred_fake = torch.tensor(0.0).to(x)
            loss_dis = torch.tensor(0.0).to(x)
            loss_adv = torch.tensor(0.0).to(x)

        # COMPOSE GEN LOSS
        global NUM_STEPS
        NUM_STEPS += 1
        beta = get_beta_kl_cyclic_annealed(
            step=NUM_STEPS,
            cycle_size=5e4,
            warmup=model.warmup // 2,
            min_beta=model.min_kl,
            max_beta=model.max_kl,
        )
        loss_gen = distance + loss_adv + beta * kl
        if model.feature_match:
            loss_gen = loss_gen + feature_matching_distance

        # OPTIMIZATION
        if False:  # model.global_step % 2 and model.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
        else:
            loss = loss_gen
            # gen_opt.zero_grad()
            # loss_gen.backward()
            # gen_opt.step()

        info.update(
            {
                "train/loss_kld": float(beta * kl),
                "train/loss_multi_spectral": float(distance),
            }
        )

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
        get_model = lambda: ae.get_autoencoder("base", hyper_cfg)
    elif hyper_cfg.model == "res-ae":
        get_model = lambda: ae.get_autoencoder("res-ae", hyper_cfg)
    elif hyper_cfg.model == "vae":
        get_model = lambda: vae.get_vae("base", hyper_cfg)
    elif hyper_cfg.model == "res-vae":
        get_model = lambda: vae.get_vae("res-vae", hyper_cfg)
    elif hyper_cfg.model == "vq-vae":
        get_model = lambda: vqvae.get_vqvae("base", hyper_cfg)
    elif hyper_cfg.model == "rave":
        get_model = lambda: rave.get_rave("base", hyper_cfg)
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
