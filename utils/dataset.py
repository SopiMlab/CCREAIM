import logging
import shutil
import tarfile
from pathlib import Path
from typing import Union

import torch
import torchaudio
from hydra.core.hydra_config import HydraConfig
from torch.nn import functional as F
from torch.utils import data

from utils import cfg_classes

log = logging.getLogger(__name__)


def prepare_dataset_on_tmp(data_tar: str, cfg: cfg_classes.BaseConfig) -> Path:
    hc = HydraConfig.get()
    tmp = Path(f"/tmp/{cfg.logging.run_id}-{hc.job.id}")
    tmp.mkdir(exist_ok=False)
    log.info(f"Copying data tar to: {tmp}")
    tmp_data_tar = shutil.copy2(data_tar, tmp)
    return tmp_data_tar


class AudioDataset(data.Dataset):
    def __init__(self, data_tar: tarfile.TarFile, seq_len: int, ext: str = "wav"):
        self.data_tar = data_tar
        self.seq_len = seq_len
        self.ext = ext
        self.sample_path_list = self.data_tar.getnames()

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, str]]:
        file_name = self.sample_path_list[index]
        file = self.data_tar.extractfile(file_name)
        waveform, _ = torchaudio.load(file, format=self.ext)  # type: ignore
        waveform = waveform.squeeze()
        padded_waveform = F.pad(waveform, (0, self.seq_len - waveform.size(0)), value=0)
        return padded_waveform.unsqueeze(0), str(self.sample_path_list[index])

    def __len__(self):
        return len(self.sample_path_list)


class FeatureDataset(data.Dataset):
    def __init__(self, data_tar: tarfile.TarFile, ext: str = "pt"):
        self.data_tar = data_tar
        self.ext = ext
        self.sample_path_list = self.data_tar.getnames()

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, str]]:
        file_name = self.sample_path_list[index]
        file = self.data_tar.extractfile(file_name)
        feature = torch.load(file, map_location="cpu")
        return feature.squeeze().T, str(self.sample_path_list[index])

    def __len__(self):
        return len(self.sample_path_list)
