import time
import timeit

import hydra
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchaudio
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf
from torchaudio.io import StreamReader, StreamWriter

from ccreaim.model import operate
from ccreaim.utils import dataset, cfg_classes
from ccreaim.utils.postprocessing import top_k_top_p_filtering
from ccreaim.utils.create_feature_dataset import create_feature_vec_from_clip

NUM_ITER = 100


# This hacky function works only with the non-augmented dataset, can be used for POC generation
def get_features_from_index(ind, dataset, seq_len):
    context, indices, a = dataset[ind // seq_len]
    assert ind == indices[ind % seq_len]
    return context[ind % seq_len]


def gen_n_sec_audio(data_non_aug, n, model_input, trf, temperature, seq_len, device):
    tgt = torch.clone(model_input)
    indices = torch.zeros(n)
    # No cacheing and outputing longer samples
    trf = trf.train()
    print(indices)
    for i in range(n):
        # Run through the forward function
        trf_out, attn_weights = trf(tgt.to(device))
        print("model_output:", trf_out.shape, attn_weights.shape)
        trf_out_filtered = top_k_top_p_filtering(trf_out[:,-1,:].squeeze() / temperature, top_k=5, top_p=0)
        print(trf_out_filtered)
        probabilities = F.softmax(trf_out_filtered)
        emb_ind = torch.multinomial(probabilities,1).item()
        indices[i] = emb_ind
        next_feature = get_features_from_index(emb_ind, data_non_aug, seq_len)
        tgt = torch.cat(
            (
                tgt,
                next_feature.unsqueeze(0).unsqueeze(0)
            ),
            dim=1
        )
        probabilities = F.softmax(trf_out, dim=-1)
        # print(probabilities.max(dim=2)[0][-1][-1])
        # print(probabilities.max(dim=2)[1][-1][-1])
    return indices


def record(q1, in_device, src, segment_length, sample_rate):
    s_in = StreamReader(src, format=in_device)
    s_in.add_basic_audio_stream(
        frames_per_chunk=segment_length, buffer_chunk_size=100, sample_rate=sample_rate
    )

    s_in_iter = s_in.stream(timeout=-1, backoff=0.1)
    for i in range(NUM_ITER):
        (chunk,) = next(s_in_iter)
        q1.put(chunk.mean(dim=1))


def process(q1, q2, model, segment_len, sample_rate, seq_len, latent_dim, data_non_aug, data_samples, device):
    # This could be worth it to rethink properly
    cur_len = 0
    secs = 4
    samples_per_inference = sample_rate * secs
    buffer_tensor = torch.zeros(samples_per_inference * 3)

    # This for-loop doesn't make sense here if segment_len < seq_len
    with torch.inference_mode():
        for i in range(NUM_ITER):
            # Fill the buffer until there's enough for the model's forward call
            while cur_len < samples_per_inference:
                new_buffer_batch = q1.get()
                mask = new_buffer_batch <= 0.025
                # model_input[mask] = 0
                buffer_tensor[cur_len : (cur_len + segment_len)] = new_buffer_batch
                cur_len += segment_len

            # Extract a view of the tensor
            model_input = buffer_tensor[0:samples_per_inference]
            print("model_init:", model_input.shape, max(model_input), min(model_input))
            model_input = create_feature_vec_from_clip(model_input, samples_per_inference, 0.2, 0.1, True)
            model_input = model_input.unsqueeze(dim=0).unsqueeze(dim=0).type(torch.FloatTensor)
            print("model_input:", model_input.shape)
            # print(model_input)

            indices = gen_n_sec_audio(data_non_aug, secs, model_input, model, 1.0, seq_len, device)
            outputs = [data_samples[int(i.item())] for i in indices]  # TODO: Now just one second samples possible to process
            print("outputs:", [(out, name) for out, name in outputs])
            print("attention:", [l.self_attn.out_proj.weight for l in model.transformer_decoder.layers])
            output = torch.cat([out for out, _ in outputs], dim=1)
            print(output)
            # print("output:", outputs.shape, name)

            # Insert to the "play"-queue
            model_output = output.cpu().t()
            print("final:", model_output.shape, max(model_output), min(model_output))
            q2.put(model_output) # .view(1, 1)

            # Reset the buffer
            buffer_tensor[0:samples_per_inference] = buffer_tensor[samples_per_inference : (2 * samples_per_inference)]
            cur_len -= samples_per_inference


def play(q2, out_device, dst, segment_length, sample_rate):
    with open("test.wav", "w") as f:
        s_out = StreamWriter(dst=dst, format=out_device)
        s_out.add_audio_stream(sample_rate, 1, format="flt")
        with s_out.open():
            times = []
            for i in range(NUM_ITER):
                chunk = q2.get().type(torch.FloatTensor)
                s_out.write_audio_chunk(0, chunk)


@hydra.main(version_base=None, config_path="cfg", config_name="live")
def main(cfg):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    src = "default"
    dst = "default"

    checkpoint = torch.load(cfg.load_model_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
    # print(hyper_cfg_schema)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    # print(conf)
    cfg.hyper = OmegaConf.merge(hyper_cfg_schema, conf)
    get_model = operate.get_model_init_function(cfg.hyper)
    model = get_model()
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    print(model)
    model.eval()

    tmp_data_root_non_aug = dataset.prepare_dataset_on_tmp(cfg.dataset_non_aug)
    data_non_aug = dataset.BankTransformerDataset(tmp_data_root_non_aug)
    tmp_data_root_samples = dataset.prepare_dataset_on_tmp(cfg.dataset_samples)
    data_samples = dataset.AudioDataset(tmp_data_root_samples, cfg.sample_rate)

    # Torch multiprocess
    ctx = mp.get_context("spawn")
    q1 = ctx.Queue()
    q2 = ctx.Queue()
    print(cfg.input_device, cfg.output_device)
    p_in = ctx.Process(
        target=record,
        args=(q1, cfg.input_device, src, cfg.segment_length, cfg.sample_rate),
    )
    p_process = ctx.Process(
        target=process, 
        args=(q1, q2, model, cfg.segment_length, cfg.sample_rate, cfg.seq_len, cfg.hyper.latent_dim, data_non_aug, data_samples, device)
    )
    p_out = ctx.Process(
        target=play,
        args=(q2, cfg.output_device, dst, cfg.segment_length, cfg.sample_rate),
    )

    p_out.start()
    p_process.start()
    p_in.start()

    print("Started")

    p_in.join()
    p_process.join()
    p_out.join()
    print("Finished")

    


if __name__ == "__main__":
    # main(input_device="alsa", src="default", output_device="alsa", dst="default")
    main()
