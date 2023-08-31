from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
import numpy
import torch
import torchaudio

# Returns both a scipy.io wavfile output and torchaudio.load output
def return_clip_from_path(path, starting_point_sec, clip_length_sec):
    torch_data, samp_rate = torchaudio.load(path)
    [Fs, x] = audioBasicIO.read_audio_file(path)
    
    if len(x.shape) != 1:
        x = numpy.average(x, axis=1).astype(x.dtype)
    if len(torch_data.shape) != 1:
        torch_data = torch.mean(torch_data, dim=0)

    torch_clip = torch_data[samp_rate*starting_point_sec : samp_rate*(starting_point_sec + clip_length_sec)]
    clip = x[samp_rate*starting_point_sec : samp_rate*(starting_point_sec + clip_length_sec)]
    return clip, torch_clip.unsqueeze(0), samp_rate