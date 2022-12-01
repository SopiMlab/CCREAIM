import time
import timeit

import torch
import torch.multiprocessing as mp
import torchaudio
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf
from torchaudio.io import StreamReader, StreamWriter

from ccreaim.model import ae as ae

NUM_ITER = 10000


def record(q, device, src, segment_length, sample_rate):
    s_in = StreamReader(src, format=device)
    s_in.add_basic_audio_stream(
        frames_per_chunk=segment_length, buffer_chunk_size=100, sample_rate=sample_rate
    )

    s_in_iter = s_in.stream(timeout=-1, backoff=0.1)
    for i in range(NUM_ITER):
        (chunk,) = next(s_in_iter)
        q.put(chunk.mean(dim=1).unsqueeze(dim=1))


def play(q, device, dst, segment_length, sample_rate):
    # model = ae.get_autoencoder()
    with open("test.wav", "w") as f:
        s_out = StreamWriter(dst=dst, format=device)
        s_out.add_audio_stream(sample_rate, 1, format="flt")
        with s_out.open():
            times = []
            for i in range(NUM_ITER):
                chunk = q.get()
                s_out.write_audio_chunk(0, chunk)


@hydra.main(version_base=None, config_path="cfg", config_name="base_live")
def main(input_device, src, output_device, dst):
    print(torch.__version__)
    print(torchaudio.__version__)

    sample_rate = 16000
    segment_length = 1024
    print(f"Sample rate: {sample_rate}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    p_in = ctx.Process(
        target=record, args=(q, input_device, src, segment_length, sample_rate)
    )
    p_out = ctx.Process(
        target=play, args=(q, output_device, dst, segment_length, sample_rate)
    )

    p_out.start()
    p_in.start()

    p_in.join()
    p_out.join()
    print("Finished")


if __name__ == "__main__":
    main(input_device="alsa", src="default", output_device="alsa", dst="default")
