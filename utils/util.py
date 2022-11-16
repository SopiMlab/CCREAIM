import io
import logging
import math
import random
import tarfile
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F

from model import ae, e2e_chunked, transformer, vae, vqvae
from utils import cfg_classes

log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)


def chop_sample(sample: torch.Tensor, sample_length: int) -> List[torch.Tensor]:
    assert len(sample.size()) == 1, "Sample is not 1 dimensional" + str(sample.size())
    chopped_samples_list: List[torch.Tensor] = []
    n_chops = len(sample) // sample_length
    for s in range(n_chops - 1):
        chopped_samples_list.append(sample[s * sample_length : (s + 1) * sample_length])
    remainder = sample[n_chops * sample_length :]
    if remainder.size(0) > 0:
        chopped_samples_list.append(remainder)
    return chopped_samples_list


def chop_dataset(in_root: str, out_tar_file_path: str, ext: str, sample_length: int):
    samples_paths = get_sample_path_list(Path(in_root), ext)
    with tarfile.open(out_tar_file_path, "a") as out_tar:
        for pth in samples_paths:
            try:
                full_sample, sample_rate = torchaudio.load(str(pth), format=ext)  # type: ignore
            except RuntimeError as e:
                log.warn(f"Could not open file, with error: {e}")
                continue

            chopped_samples = chop_sample(full_sample.squeeze(), sample_length)
            for i, cs in enumerate(chopped_samples):
                out_name = str(pth.stem) + f"_{i:03d}" + ".wav"
                with io.BytesIO() as buffer:
                    torchaudio.save(  # type: ignore
                        buffer,
                        cs.unsqueeze(0),
                        sample_rate,
                        encoding="PCM_F",
                        bits_per_sample=32,
                    )
                    out_info = tarfile.TarInfo(name=out_name)
                    out_info.size = buffer.getbuffer().nbytes
                    out_tar.addfile(out_info, buffer)


def get_sample_path_list(data_root: Path, ext: str = "mp3") -> List[Path]:
    return list(data_root.rglob(f"*.{ext}"))


# Calculates padding for nn.Conv1d-layers to achieve l_out=ceil(l_in/stride)
def conf_same_padding_calc(length: int, stride: int, kernel_size: int):
    out_length = math.ceil(float(length) / float(stride))

    if length % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (length % stride), 0)

    return math.ceil(pad / 2), out_length


# Calculates padding and output_padding for nn.ConvTranspose1d to get preferred length_out with minimal output_padding
def conf_same_padding_calc_t(
    length_in: int, length_out: int, stride: int, kernel_size: int
):
    padding = math.ceil(((length_in - 1) * stride - length_out + kernel_size) / 2)
    output_padding = length_out - ((length_in - 1) * stride - 2 * padding + kernel_size)
    return padding, output_padding


# Returns the path to the directory where a model is exported to/imported from according
# to configuration in cfg, as well as the base name of the model.
def get_model_path(cfg: cfg_classes.BaseConfig):
    exp_path = Path(cfg.logging.model_checkpoints)
    model_name = f"{cfg.hyper.model}_seqlen-{cfg.hyper.seq_len}_bs-{cfg.hyper.batch_size}_lr-{cfg.hyper.learning_rate}_seed-{cfg.hyper.seed}"
    return exp_path, model_name


# Spectral loss
class STFTValues:
    def __init__(self, n_bins: int, hop_length: int, window_size: int):
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.window_size = window_size


def norm(x: torch.Tensor):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def spec(seq: torch.Tensor, stft_val: STFTValues):
    return torch.norm(
        torch.stft(
            seq,
            stft_val.n_bins,
            stft_val.hop_length,
            win_length=stft_val.window_size,
            window=torch.hann_window(stft_val.window_size, device=seq.device),
        ),
        p=2,
        dim=-1,
    )


def multispectral_loss(
    seq: torch.Tensor, pred: torch.Tensor, cfg: cfg_classes.BaseConfig
) -> torch.Tensor:
    losses = torch.zeros(*seq.size()[:-1], device=seq.device)
    if losses.ndim == 1:
        losses = losses.unsqueeze(-1)
        seq = seq.unsqueeze(1)
        pred = pred.unsqueeze(1)
    args = (
        cfg.hyper.spectral_loss.stft_bins,
        cfg.hyper.spectral_loss.stft_hop_length,
        cfg.hyper.spectral_loss.stft_window_size,
    )
    for n_bins, hop_length, window_size in zip(*args):
        stft_val = STFTValues(n_bins, hop_length, window_size)
        for i in range(losses.size(-1)):
            spec_in = spec(seq[:, i].squeeze(), stft_val)
            spec_out = spec(pred[:, i].squeeze(), stft_val)
            losses[:, i] = norm(spec_in - spec_out)
    return losses


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
        pred = model(seq, pad_mask, device)
        tgt = seq[:, 1:, :]
        tgt_pad_mask = pad_mask[:, 1:]
        mse = F.mse_loss(pred, tgt, reduction="none")
        mse[tgt_pad_mask] = 0
        mse = mse.mean()
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = multispectral_loss(tgt, pred, cfg)
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
        multi_spec = multispectral_loss(seq, pred, cfg)
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
        pred = model(seq)
        mse = F.mse_loss(pred, seq)
        spec_weight = cfg.hyper.spectral_loss.weight
        multi_spec = multispectral_loss(seq, pred, cfg)
        multi_spec = multi_spec.mean()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_spectral": float(spec_weight * multi_spec.item()),
            }
        )
        losee = mse + spec_weight * multi_spec
    return loss, pred, info
