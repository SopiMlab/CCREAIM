import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils import data

from utils import dataset


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
        full_sample, sample_rate = torchaudio.load(str(pth))
        chopped_samples = chop_sample(full_sample, sample_length)
        for i, cs in enumerate(chopped_samples):
            out_path = Path(out_root) / Path(
                str(pth.stem) + f"_{i:03d}" + str(pth.suffix)
            )
            torchaudio.save(out_path, cs, sample_rate)


def get_sample_path_list(data_root: Path, ext: str = "mp3") -> List[Path]:
    return list(data_root.rglob(f"*.{ext}"))
