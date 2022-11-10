from pathlib import Path
from typing import Union

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils import data

import utils.util as util


class AudioDataset(data.Dataset):
    def __init__(self, data_root: Path, seq_len: int, ext: str = "wav"):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.ext = ext
        self.sample_path_list = util.get_sample_path_list(self.data_root, self.ext)

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, str]]:
        waveform, _ = torchaudio.load(  # type: ignore
            str(self.sample_path_list[index]), format=self.ext
        )
        waveform = waveform.squeeze()
        padded_waveform = F.pad(waveform, (0, self.seq_len - waveform.size(0)), value=0)
        return padded_waveform.unsqueeze(0), str(self.sample_path_list[index])

    def __len__(self):
        return len(self.sample_path_list)


class FeatureDataset(data.Dataset):
    def __init__(self, data_root: Path, ext: str = "pt"):
        self.data_root = Path(data_root)
        self.ext = ext
        self.sample_path_list = util.get_sample_path_list(self.data_root, self.ext)

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, str]]:
        feature = torch.load(str(self.sample_path_list[index]), map_location="cpu")
        return feature.squeeze().T, str(self.sample_path_list[index])

    def __len__(self):
        return len(self.sample_path_list)
