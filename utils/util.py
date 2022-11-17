import io
import logging
import math
import random
import tarfile
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio

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


def save_model_prediction(model_name: str, pred: torch.Tensor, save_path: Path) -> None:
    if model_name == "transformer":
        torch.save(pred, save_path)
    elif model_name == "e2e-chunked":
        torchaudio.save(  # type: ignore
            save_path,
            pred.flatten().unsqueeze(0),
            16000,
            encoding="PCM_F",
            bits_per_sample=32,
        )
    else:
        torchaudio.save(  # type: ignore
            save_path, pred, 16000, encoding="PCM_F", bits_per_sample=32
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
            spec_in = spec(seq[:, i], stft_val)
            spec_out = spec(pred[:, i], stft_val)
            losses[:, i] = norm(spec_in - spec_out)
    return losses
