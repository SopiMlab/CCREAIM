import io
import os
import tarfile
from pathlib import Path

import numpy
import torch
import torchaudio
from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
from scipy.io import wavfile


def create_feature_vec_from_clip(x, Fs, frame_size, frame_step, deltas):
    F, f_names = ShortTermFeatures.feature_extraction(
        x, Fs, frame_size * Fs, frame_step * Fs, deltas=deltas
    )
    features = torch.tensor(F).view(-1)
    return features


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
    frame_size = 0.2
    frame_step = 0.1
    deltas = True

    sample_dir = "/scratch/other/sopi/CCREAIM/datasets/maestro_bank_data_samples"
    training_data_tar = "/scratch/other/sopi/CCREAIM/datasets/maestro_bank_training.tar"
    bank_data_tar = "/scratch/other/sopi/CCREAIM/datasets/maestro_bank.tar"

    data_dir = "/scratch/other/sopi/CCREAIM/datasets/maestro_bank_data"
    sample_paths = iter(sorted(os.listdir(data_dir)))

    total_samples_loaded = 0
    with tarfile.open(training_data_tar, "a") as training_out_tar:
        with tarfile.open(bank_data_tar, "a") as bank_out_tar:
            while total_samples_loaded < vocab_size:
                sample_path = next(sample_paths)
                torch_data, samp_rate = torchaudio.load(f"{data_dir}/{sample_path}")
                [Fs, x] = audioBasicIO.read_audio_file(f"{data_dir}/{sample_path}")
                if len(x.shape) != 1:
                    x = numpy.average(x, axis=1).astype(x.dtype)
                if len(torch_data) != 1:
                    torch_data = torch.mean(torch_data, dim=0)
                num_contexts = len(x) // (
                    Fs * sample_len_seconds * transformer_ctxt_len
                )
                for j in range(num_contexts):
                    bank_context = []
                    training_context = []
                    for i in range(transformer_ctxt_len):
                        name = f"ind_{total_samples_loaded:05d}_{Path(sample_path).stem}_context_{j:d}_sample{i:d}"
                        pointer_cur = (
                            (j * transformer_ctxt_len + i) * sample_len_seconds * Fs
                        )
                        sample_length = sample_len_seconds * Fs
                        sample = x[pointer_cur : (pointer_cur + sample_length)]
                        torch_sample = torch_data[
                            pointer_cur : (pointer_cur + sample_length)
                        ]
                        features = create_feature_vec_from_clip(
                            sample, Fs, frame_size, frame_step, deltas
                        )
                        bank_context.append(
                            (
                                features,
                                torch_sample,
                                total_samples_loaded,
                                samp_rate,
                                f"{name}.pt",
                            )
                        )
                        training_context.append(
                            (features, total_samples_loaded, f"{name}.pt")
                        )
                        total_samples_loaded += 1
                        if (total_samples_loaded - 1) % 300 == 0:
                            print(f"saving a sample: {name}.wav")
                        # torchaudio.save(f"{sample_dir}/{name}.wav", torch_sample.unsqueeze(0), samp_rate, encoding="PCM_F", bits_per_sample=32, format="wav")
                    save_object_to_tar(training_out_tar, training_context, f"{name}.pt")
                    save_object_to_tar(bank_out_tar, bank_context, f"{name}.pt")
                    if total_samples_loaded >= vocab_size:
                        break
                print(total_samples_loaded)


if __name__ == "__main__":
    main()
