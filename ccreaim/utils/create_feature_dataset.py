import io
import os
import tarfile
from pathlib import Path

import numpy
import torch
import torchaudio
from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
from scipy.io import wavfile
from torchaudio_augmentations import *

"""
This file can be used to create the training dataset required for the bank-classifier model.
For other necessary scripts to run, check "create_sample_bank.py".
"""


# Extract features from any length of audio
# Check out
# https://github.com/tyiannak/pyAudioAnalysis/wiki
# args:
#   x: audio data, audioBasicIO.read_audio_file() output
#   Fs: sample rate, audioBasicIO.read_audio_file() output
#   frame_size: how big of a window to extract the features from
#   frame_step: how big steps do we take for extracting the features
#   deltas: boolean, do we include the first derivative of the features as well
def create_feature_vec_from_clip(x, Fs, frame_size, frame_step, deltas):
    F, f_names = ShortTermFeatures.feature_extraction(
        x, Fs, frame_size * Fs, frame_step * Fs, deltas=deltas
    )
    features = torch.tensor(F).view(-1)
    return features


# Helper function to save any python data to a file inside a tar archive,
# args:
#   out_tar: path to the archive
#   obj: python object to save
#   file_name: name of the destination file inside the archive
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


# Produce augmentations from an audio sample, input should be torchaudio.load() output
# num_datapoints is the length of the audio sample, basically sample_rate*sample_length_in_seconds
# Check out
# https://github.com/Spijkervet/torchaudio-augmentations
def data_augmentation(sample, num_datapoints, sample_rate, num_augments):
    transforms = [
        RandomResizedCrop(n_samples=num_datapoints),
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
        # RandomApply([Gain()], p=0.2), # removed since gain can make the model not care about sudden changes in amplitude, which is unmusical
        RandomApply([HighLowPass(sample_rate=sample_rate)], p=0.2),
        RandomApply([Delay(sample_rate=sample_rate)], p=0.5),
        RandomApply(
           [PitchShift(n_samples=num_datapoints, sample_rate=sample_rate)], p=0.4 # removed since it's clearly important to do distinctions between different pitches
        ),
        RandomApply([Reverb(sample_rate=sample_rate)], p=0.3),
    ]
    transform = ComposeMany(transforms=transforms, num_augmented_samples=num_augments)
    return transform(sample)


# Given a sample from torchaudio.load and relevant parameters,
# returns the features from all the augmented versions of the
# sample
def get_augmented_features(
    sample, num_datapoints, sample_rate, num_augments, frame_size, frame_step, deltas
):
    augmented_samples = data_augmentation(
        sample, num_datapoints, sample_rate, num_augments
    )
    features_list = []
    for i in range(num_augments):
        # Do this awful temporary save since the formats of pyAudioAnalysis (which uses scipy.io:s wavfile) and torchaudio
        # do not match, augmentation uses torchaudio format and feature extraction uses scipy.io.wavfile format
        # There's probably a much cleaner way to do this
        torchaudio.save(
            f"/tmp/aug{i:05d}.wav",
            augmented_samples[i, :, :],
            sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
            format="wav",
        )
        [Fs, x] = audioBasicIO.read_audio_file(f"/tmp/aug{i:05d}.wav")
        os.remove(f"/tmp/aug{i:05d}.wav")
        features_list.append(
            create_feature_vec_from_clip(x, Fs, frame_size, frame_step, deltas)
        )
    return features_list


def main():
    ###########################################################################
    ################################
    ###### Modifiable inputs #######
    ################################
    # Dataset/transformer input/output defining parameters
    vocab_size = 8192
    sample_len_seconds = 1
    transformer_ctxt_len = 8

    # Feature extraction parameters
    frame_size = 0.2
    frame_step = 0.1
    deltas = True

    # Number of (stochasticly) augmented samples to produce
    num_augments = 10

    # If you want to save some of the samples, where should they go
    sample_dir = "/scratch/other/sopi/CCREAIM/datasets/samples_bank_data_samples"

    # Path to the transformer's training data
    training_data_tar = (
        "/scratch/other/sopi/CCREAIM/datasets/samples_sound_bank_training_aug.tar"
    )

    # Where to find the data, from which the dataset will be extracted
    # This directory should contain at least vocab_size*sample_len_seconds of audio, otherwise the
    # script eventually fails at next(sample_paths)
    data_dir = "/scratch/other/sopi/CCREAIM/datasets/samples"
    ###########################################################################

    # Make an iterator out of all the names of files in data_dir
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and not f.startswith('.') and f.endswith('.wav')]
    print(len(files))
    sample_paths = iter(sorted(files))

    # Keep track of the number of total samples loaded
    id = 0
    total_samples_loaded = 0
    with tarfile.open(training_data_tar, "a") as training_out_tar:
        # Continue until we have loaded a total of vocab_size audio samples
        while total_samples_loaded < vocab_size:
            print("id:", id)
            id+=1
            # Load both a scipy.io.wavfile and torchaudio.load formats
            sample_path = next(sample_paths)
            torch_data, samp_rate = torchaudio.load(f"{data_dir}/{sample_path}")
            [Fs, x] = audioBasicIO.read_audio_file(f"{data_dir}/{sample_path}")

            # If there are multiple channels, convert to mono by taking the average of the channels
            if len(x.shape) != 1:
                x = numpy.average(x, axis=1).astype(x.dtype)
            if len(torch_data.shape) != 1:
                torch_data = torch.mean(torch_data, dim=0)

            # Iterate for the number of transformer contexts, aka transformer training samples (not including augmentation),
            # in the whole audio file (rounded down)
            num_contexts = len(x) // (Fs * sample_len_seconds * transformer_ctxt_len)
            for j in range(num_contexts):
                # If we manage to get here and the dataset is already completed, do nothing (should be unnecessary)
                if total_samples_loaded >= vocab_size:
                    break

                # Store the context data in a list
                training_context = []

                # one context per augmentation
                augmented_contexts = [[] for x in range(num_augments)]

                # Iterate through 'transformer_ctxt_len' samples to store their features in the context
                for i in range(transformer_ctxt_len):
                    # Sample specific name
                    name = f"ind_{total_samples_loaded:05d}_{Path(sample_path).stem}_context_{j:d}_sample{i:d}"

                    # Retrieve a clip of data from the relevant location
                    pointer_cur = (
                        (j * transformer_ctxt_len + i) * sample_len_seconds * Fs
                    )
                    sample_length = sample_len_seconds * Fs
                    sample = x[pointer_cur : (pointer_cur + sample_length)]
                    torch_sample = torch_data[
                        pointer_cur : (pointer_cur + sample_length)
                    ]

                    # Extract features from the clip
                    features = create_feature_vec_from_clip(
                        sample, Fs, frame_size, frame_step, deltas
                    )

                    # Extract features from augmented versions of the clip
                    if num_augments > 0:
                        aug_features = get_augmented_features(
                            torch_sample.unsqueeze(0),
                            samp_rate * sample_len_seconds,
                            samp_rate,
                            num_augments,
                            frame_size,
                            frame_step,
                            deltas,
                        )

                    # Insert the context data into the lists
                    for k in range(num_augments):
                        augmented_contexts[k].append(
                            (
                                aug_features[k],
                                total_samples_loaded,
                                f"aug{k:02d}_{name}.pt",
                            )
                        )
                    training_context.append(
                        (features, total_samples_loaded, f"{name}.pt")
                    )

                    # One more sample has been processed
                    total_samples_loaded += 1

                    # Every 300 audio samples save that sample into {sample_dir}/
                    if (total_samples_loaded - 1) % 300 == 0:
                        print(f"saving a sample: {name}.wav")
                        # torchaudio.save(f"{sample_dir}/{name}.wav", torch_sample.unsqueeze(0), samp_rate, encoding="PCM_F", bits_per_sample=32, format="wav")
                print("total samples:", total_samples_loaded)
                # Save training data into training_data_tar
                save_object_to_tar(
                    training_out_tar, training_context, f"sample_{id}_context_{j}.pt"
                )
                for k in range(num_augments):
                    aug_context = augmented_contexts[k]
                    save_object_to_tar(
                        training_out_tar, aug_context, f"sample_{id}_context_{j}_aug{k}.pt"
                    )
            # After going through a
            print(total_samples_loaded)


if __name__ == "__main__":
    main()
