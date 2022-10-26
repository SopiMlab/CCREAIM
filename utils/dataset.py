from pathlib import Path
from typing import Generator, List

import torchaudio
from torch.utils import data

import utils.util as util


class AudioDataset(data.Dataset):
    def __init__(self, data_root: str, ext: str = "mp3"):
        self.data_root = Path(data_root)
        self.ext = ext
        self.sample_path_list = util.get_sample_path_list(self.data_root, self.ext)

    def __getitem__(self, index: int):
        waveform, sample_rate = torchaudio.load(
            str(self.sample_path_list[index]), format=self.ext
        )

        return waveform.squeeze()

    def __len__(self):
        return len(self.sample_path_list)
