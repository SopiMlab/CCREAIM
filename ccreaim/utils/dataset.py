import logging
import shutil
import tarfile
from pathlib import Path
from time import sleep
from typing import Union

import torch
import torchaudio
from hydra.core.hydra_config import HydraConfig
from torch.nn import functional as F
from torch.utils import data

from ..utils import util
from ..utils.cfg_classes import LoggingConfig

log = logging.getLogger(__name__)


def prepare_dataset_on_tmp(data_tar: str, seq_len: int) -> Path:
    tmp_preparing = Path(f"/tmp/chopped_{seq_len}_preparing")
    tmp = Path(f"/tmp/chopped_{seq_len}")
    while tmp_preparing.exists():
        log.info(
            f"{str(tmp_preparing)} exists on the filesystem, waiting for data preparation"
        )
        sleep(10)
    if tmp.exists():
        log.info(f"Pre-existing dataset found at {str(tmp)}, skipping data preparation")
        return tmp
    else:
        tmp_preparing.mkdir(exist_ok=False)
        log.info(f"Copying data tar to: {tmp_preparing}")
        tmp_data_tar_path = shutil.copy2(data_tar, tmp_preparing)
        with tarfile.open(tmp_data_tar_path, "r") as tmp_data_tar:
            log.info(f"Extracting files")
            tmp_data_tar.extractall(tmp_preparing)
        Path(tmp_data_tar_path).unlink()
        tmp_preparing.rename(tmp)
        return tmp


class AudioDataset(data.Dataset):
    def __init__(self, data_root: Path, seq_len: int, ext: str = "wav"):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.ext = ext
        self.sample_path_list = util.get_sample_path_list(self.data_root, self.ext)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        file_name = str(self.sample_path_list[index])
        waveform, _ = torchaudio.load(file_name, format=self.ext)  # type: ignore
        waveform = waveform.squeeze()
        padded_waveform = F.pad(waveform, (0, self.seq_len - waveform.size(0)), value=0)
        return padded_waveform.unsqueeze(0), file_name

    def __len__(self):
        return len(self.sample_path_list)


class ChunkedAudioDataset(data.Dataset):
    def __init__(self, data_root: Path, seq_len: int, seq_num: int, ext: str = "wav"):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.ext = ext
        self.seq_num = seq_num
        self.sample_path_list = util.get_sample_path_list(self.data_root, self.ext)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        file_name = str(self.sample_path_list[index])
        waveform, _ = torchaudio.load(file_name, format=self.ext)  # type: ignore
        waveform = waveform.squeeze()
        seq_list = util.chop_sample(waveform, self.seq_len)
        seq_list[-1] = F.pad(
            seq_list[-1], (0, self.seq_len - seq_list[-1].size(0)), value=0
        )
        seq = torch.stack(seq_list)
        padded_seq = F.pad(seq, (0, 0, 0, self.seq_num - seq.size(0)), value=0)
        pad_mask = torch.full((self.seq_num,), False)
        pad_mask[seq.size(0) + 1 :] = True
        return padded_seq, file_name, pad_mask

    def __len__(self):
        return len(self.sample_path_list)


class FeatureDataset(data.Dataset):
    def __init__(self, data_root: Path, ext: str = "pt"):
        self.data_root = data_root
        self.ext = ext
        self.sample_path_list = util.get_sample_path_list(self.data_root, self.ext)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        file_name = str(self.sample_path_list[index])
        feature = torch.load(file_name, map_location="cpu")
        return feature.squeeze().T, file_name

    def __len__(self):
        return len(self.sample_path_list)
