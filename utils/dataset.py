from pathlib import Path
from typing import Generator, List

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

    def __getitem__(self, index: int):
        waveform, _ = torchaudio.load(
            str(self.sample_path_list[index]), format=self.ext
        )
        waveform = waveform.squeeze()
        padded_waveform = F.pad(waveform, (0, self.seq_len - waveform.size(0)), value=0)
        return padded_waveform.unsqueeze(0)

    def __len__(self):
        return len(self.sample_path_list)
