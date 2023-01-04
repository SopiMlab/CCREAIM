import sys

sys.path.append("..")
from pathlib import Path

from ccreaim.utils import util

test_dataset = True
sample_length = 4096
ext = "mp3"

input_root = (
    "/scratch/other/sopi/CCREAIM/datasets/test/in"
    if test_dataset
    else "/scratch/other/sopi/CCREAIM/datasets/magna-tag-a-tune"
)
output_root = (
    "/scratch/other/sopi/CCREAIM/datasets/test/out"
    if test_dataset
    else "/scratch/other/sopi/CCREAIM/datasets"
)

util.chop_dataset(
    Path(input_root),
    Path(output_root, f"chopped_{sample_length}.tar"),
    ext,
    sample_length,
)
