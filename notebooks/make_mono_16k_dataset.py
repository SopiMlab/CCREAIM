import sys

sys.path.append("..")
from pathlib import Path

import torchaudio
from torchaudio import functional as F

from ccreaim.utils import util

test_dataset = True
ext = "wav"

input_root = (
    "/scratch/other/sopi/CCREAIM/datasets/test/in_maestro"
    if test_dataset
    else "/scratch/other/sopi/CCREAIM/datasets/magna-tag-a-tune"
)
output_root = (
    "/scratch/other/sopi/CCREAIM/datasets/test/out_maestro"
    if test_dataset
    else "/scratch/other/sopi/CCREAIM/datasets"
)

sample_paths = util.get_sample_path_list(Path(input_root), ext)
for pth in sample_paths:
    waveform, sample_rate = torchaudio.load(str(pth))
    resampled_waveform = F.resample(waveform, sample_rate, 16000)
    mono_resampled_waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(
        str(Path(output_root, pth.name)),
        mono_resampled_waveform,
        16000,
        encoding="PCM_F",
        bits_per_sample=32,
        format="wav",
    )
