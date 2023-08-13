import logging
import shutil
import tarfile
from pathlib import Path
from time import sleep
from typing import Union

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils import data

from ..utils import util

log = logging.getLogger(__name__)


def prepare_dataset_on_tmp(data_tar: str) -> Path:
    dir_name = Path(data_tar).stem
    tmp_preparing = Path(f"/tmp/{dir_name}_preparing")
    tmp = Path(f"/tmp/{dir_name}")
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
            log.info(f"Extracting files from {data_tar} to {str(tmp_preparing)}")
            tmp_data_tar.extractall(tmp_preparing)
        Path(tmp_data_tar_path).unlink()
        tmp_preparing.rename(tmp)
        log.info(
            f"Data preparation complete, renamed {str(tmp_preparing)} => {str(tmp)}"
        )
        return tmp


class AudioDataset(data.Dataset):
    def __init__(self, data_root: Path, seq_len: int, ext: str = "wav"):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.ext = ext
        self.sample_path_list = util.get_sample_path_list_orig(self.data_root, self.ext)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        file_name = str(self.sample_path_list[index])
        waveform, _ = torchaudio.load(file_name, format=self.ext)  # type: ignore
        waveform = waveform.squeeze()
        padded_waveform = F.pad(waveform, (0, self.seq_len - waveform.size(0)), value=0)
        return padded_waveform.unsqueeze(0), file_name

    def __len__(self):
        return len(self.sample_path_list)


class BankTransformerDataset(data.Dataset):
    def __init__(self, data_root: Path, ext: str = "pt"):
        self.data_root = data_root
        self.ext = ext
        # Change between get_sample_path_list and get_sample_path_list_orig when using model trained on samples and maestro, respectively 
        self.item_path_list = util.get_sample_path_list_orig(self.data_root, self.ext)

    def __getitem__(
        self, index: int
    ) -> list[tuple[torch.Tensor, torch.Tensor, int, int, str]]:
        file_name = str(self.item_path_list[index])
        data = torch.load(file_name, map_location="cpu")
        context = torch.zeros([len(data), list(data[0][0].size())[0]])
        indices = torch.zeros([len(data)])
        for i in range(len(data)):
            context[i, :] = data[i][0]
            indices[i] = data[i][1]
        return context, indices, data

    def __len__(self):
        return len(self.item_path_list)
