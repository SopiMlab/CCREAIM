import io
import os
import tarfile
from pathlib import Path

import torch
import torchaudio

# This file contains a lot of repeated code from "create_feature_dataset.py", where the
# code is well commented as well


def save_object_to_tar(out_tar, obj, file_name):
    with io.BytesIO() as buffer:
        try:
            torch.save(obj, buffer)
            buffer.seek(0)
            out_info = tarfile.TarInfo(name=file_name)
            out_info.size = buffer.getbuffer().nbytes
            out_tar.addfile(tarinfo=out_info, fileobj=buffer)
        except Exception as e:
            print(e)


def main():
    vocab_size = 8192
    sample_len_seconds = 1
    transformer_ctxt_len = 8

    sample_data_tar = "/scratch/other/sopi/CCREAIM/datasets/sounds_bank_samples.tar"

    data_dir = "/scratch/other/sopi/CCREAIM/datasets/samples"
    sample_paths = iter(sorted(os.listdir(data_dir)))

    total_samples_loaded = 0

    print(sample_data_tar)

    with tarfile.open(sample_data_tar, "a") as out_tar:
        while total_samples_loaded < vocab_size:
            sample_path = next(sample_paths)
            torch_data, samp_rate = torchaudio.load(f"{data_dir}/{sample_path}")
            if len(torch_data.shape) != 1:
                torch_data = torch.mean(torch_data, dim=0)
            num_contexts = len(torch_data) // (
                samp_rate * sample_len_seconds * transformer_ctxt_len
            )
            for j in range(num_contexts):
                if total_samples_loaded >= vocab_size:
                    break
                for i in range(transformer_ctxt_len):
                    name = f"ind_{total_samples_loaded:05d}_{Path(sample_path).stem}_context_{j:d}_sample{i:d}"
                    pointer_cur = (
                        (j * transformer_ctxt_len + i) * sample_len_seconds * samp_rate
                    )
                    sample_length = sample_len_seconds * samp_rate
                    torch_sample = torch_data[
                        pointer_cur : (pointer_cur + sample_length)
                    ]
                    with io.BytesIO() as buffer:
                        try:
                            torchaudio.save(  # type: ignore
                                buffer,
                                torch_sample.unsqueeze(0),
                                samp_rate,
                                encoding="PCM_F",
                                bits_per_sample=32,
                                format="wav",
                            )
                            buffer.seek(0)  # go to the beginning for reading the buffer
                            out_info = tarfile.TarInfo(name=name + ".wav")
                            out_info.size = buffer.getbuffer().nbytes
                            out_tar.addfile(tarinfo=out_info, fileobj=buffer)
                        except Exception as e:
                            print(e)
                    total_samples_loaded += 1
                print(total_samples_loaded)


if __name__ == "__main__":
    main()
