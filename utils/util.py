import logging
import math
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F

from model import ae, transformer, vae, vqvae
from utils import cfg_classes

log = logging.getLogger(__file__)


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
    chopped_samples_list.append(sample[n_chops * sample_length :])
    return chopped_samples_list


def chop_dataset(in_root: str, out_root: str, ext: str, sample_length: int):
    samples_paths = get_sample_path_list(Path(in_root), ext)
    for pth in samples_paths:
        try:
            full_sample, sample_rate = torchaudio.load(str(pth), format=ext)  # type: ignore
        except RuntimeError as e:
            log.warn(f"Could not open file, with error: {e}")
            continue

        chopped_samples = chop_sample(full_sample.squeeze(), sample_length)
        for i, cs in enumerate(chopped_samples):
            out_path = Path(out_root) / Path(str(pth.stem) + f"_{i:03d}" + ".wav")
            torchaudio.save(  # type: ignore
                out_path,
                cs.unsqueeze(0),
                sample_rate,
                encoding="PCM_F",
                bits_per_sample=32,
            )


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


def step(
    model: torch.nn.Module, seq: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}
    if isinstance(model, transformer.Transformer):
        src = seq[:, :-1, :]
        tgt = seq[:, 1:, :]
        tgt_mask = model.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        pred = model(src, tgt, tgt_mask)
        loss = F.mse_loss(pred, tgt)
    elif isinstance(model, vae.VAE):
        pred, mu, sigma = model(seq)
        mse = F.mse_loss(pred, seq)
        kld = -0.5 * (1 + torch.log(sigma**2) - mu**2 - sigma**2).sum()
        info.update(
            {
                "loss_mse": float(mse.item()),
                "loss_kld": float((model.kld_weight * kld).item()),
            }
        )
        loss = mse + model.kld_weight * kld
    else:
        pred = model(seq)
        loss = F.mse_loss(pred, seq)
    return loss, pred, info
