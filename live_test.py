import time
import timeit

import hydra
import torch
import torch.multiprocessing as mp
import torchaudio
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf
from torchaudio.io import StreamReader, StreamWriter

from ccreaim.model import operate
from ccreaim.utils import cfg_classes

NUM_ITER = 10000


def record(q1, in_device, src, segment_length, sample_rate):
    s_in = StreamReader(src, format=in_device)
    s_in.add_basic_audio_stream(
        frames_per_chunk=segment_length, buffer_chunk_size=100, sample_rate=sample_rate
    )

    s_in_iter = s_in.stream(timeout=-1, backoff=0.1)
    for i in range(NUM_ITER):
        (chunk,) = next(s_in_iter)
        q1.put(chunk.mean(dim=1))


def process(q1, q2, model, segment_len, seq_len, device):
    # This could be worth it to rethink properly
    cur_len = 0
    buffer_tensor = torch.zeros(max(2 * segment_len, 2 * seq_len))

    # This for-loop doesn't make sense here if segment_len < seq_len
    with torch.no_grad():
        for i in range(NUM_ITER):
            # Fill the buffer until there's enough for the model's forward call
            while cur_len < seq_len:
                buffer_tensor[cur_len : (cur_len + segment_len)] = q1.get()
                cur_len += segment_len

            # Extract a view of the tensor
            model_input = buffer_tensor[0:seq_len]

            print("Running the model")
            # Run through the forward function
            model_output = model(
                model_input.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
            )

            # Insert to the "play"-queue
            model_output = model_output[0].cpu()
            q2.put(model_output.view(seq_len, 1))

            # Reset the buffer
            buffer_tensor[0:seq_len] = buffer_tensor[seq_len : (2 * seq_len)]
            cur_len -= seq_len


def play(q2, out_device, dst, segment_length, sample_rate):
    with open("test.wav", "w") as f:
        s_out = StreamWriter(dst=dst, format=out_device)
        s_out.add_audio_stream(sample_rate, 1, format="flt")
        with s_out.open():
            times = []
            for i in range(NUM_ITER):
                chunk = q2.get()
                s_out.write_audio_chunk(0, chunk)


@hydra.main(version_base=None, config_path="cfg", config_name="live")
def main(cfg):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    src = "default"
    dst = "default"

    checkpoint = torch.load(cfg.load_model_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    cfg.hyper = OmegaConf.merge(hyper_cfg_schema, conf)
    get_model = operate.get_model_init_function(cfg.hyper)
    model = get_model()
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    model.eval()

    # Torch multiprocess
    ctx = mp.get_context("spawn")
    q1 = ctx.Queue()
    q2 = ctx.Queue()
    p_in = ctx.Process(
        target=record,
        args=(q1, cfg.input_device, src, cfg.segment_length, cfg.sample_rate),
    )
    p_process = ctx.Process(
        target=process, args=(q1, q2, model, cfg.segment_length, cfg.seq_len, device)
    )
    p_out = ctx.Process(
        target=play,
        args=(q2, cfg.output_device, dst, cfg.segment_length, cfg.sample_rate),
    )

    p_out.start()
    p_process.start()
    p_in.start()

    p_in.join()
    p_process.join()
    p_out.join()
    print("Finished")


if __name__ == "__main__":
    # main(input_device="alsa", src="default", output_device="alsa", dst="default")
    main()
