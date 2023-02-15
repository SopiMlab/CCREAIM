import sys

sys.path.append("..")
import torch
import torch.utils.data
import torchaudio
from omegaconf import OmegaConf
from torch.nn import functional as F

from ccreaim.model import operate
from ccreaim.utils import audio_tools, cfg_classes, dataset, util

data_tar = "/scratch/other/sopi/CCREAIM/datasets/test/out/chopped_65536.tar"


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


hyper_cfg = cfg_classes.HyperConfig(
    model="e2e-chunked_res-vqvae",
    pre_trained_vqvae_path="/home/baffyb1/CCREAIM/CCREAIM/logs/2023-01-04/res-vqvae_train_12-46-18/0/checkpoints/res-vqvae_seqlen-8192_bs-128_lr-0.0001_seed-0_final.pt",
    pre_trained_decoder_only_path="/home/baffyb1/CCREAIM/CCREAIM/logs/2023-01-20/transformer-decoder-only_train_13-04-48/0/checkpoints/transformer-decoder-only_seqlen-2048_bs-16_lr-0.0001_seed-0_final.pt",
    latent_dim=8,
    num_seq=8,
    seq_cat=True,
    epochs=1,
    lr_scheduler_gamma=1.0,
    seq_len=8192,
    batch_size=8,
    learning_rate=0.0001,
    seed=0,
    vqvae=cfg_classes.VQVAEConfig(
        num_embeddings=512,
        reset_patience=0.0,
        beta=0.25,
    ),
    res_ae=cfg_classes.ResAeConfig(
        levels=1,
        downs_t=[5],
        strides_t=[2],
        input_emb_width=1,
        block_width=32,
        block_depth=3,
        block_m_conv=1.0,
        block_dilation_growth_rate=3,
        block_dilation_cycle=None,
    ),
    transformer=cfg_classes.TransformerConfig(
        num_heads_latent_dimension_div=2,
        num_dec_layers=2,
        num_enc_layers=0,
        autoregressive_loss_weight=1.0,
        linear_map=True,
    ),
)

hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
get_model = operate.get_model_init_function(hyper_cfg)
model = get_model()
model = model.to(device)
model = model.eval()

tmp_data_root = dataset.prepare_dataset_on_tmp(data_tar=data_tar)
data = dataset.ChunkedAudioDataset(tmp_data_root, 8, hyper_cfg.seq_len)
dl = iter(
    torch.utils.data.DataLoader(data, batch_size=hyper_cfg.batch_size, shuffle=False)
)

sample, _, ids = next(dl)
print(sample.size())

sample_save = sample.flatten(1, 2).cpu()
print(sample_save.size())
for i in range(hyper_cfg.batch_size):
    torchaudio.save(
        f"/scratch/other/sopi/CCREAIM/datasets/test/out/e2e_test_original_{i}.wav",
        sample_save[i].unsqueeze(0),
        16000,
        encoding="PCM_F",
        bits_per_sample=32,
        format="wav",
    )


with torch.inference_mode():
    src_batch = sample.to(device)

    feed_in = 4
    gen_size = 2
    for i in range(4):
        src = src_batch[0]
        print(src.size())
        src = src[i * hyper_cfg.seq_len : (i + feed_in) * hyper_cfg.seq_len, :]
        print(src.size())

        gen = model.generate(
            src.unsqueeze(0), feed_in_chunks=feed_in, gen_chunks=gen_size
        )
        print(gen.size())
        gen_save = gen.flatten(1, 2).cpu()[0]
        torchaudio.save(
            f"/scratch/other/sopi/CCREAIM/datasets/test/out/e2e_test_generation_{i}.wav",
            gen_save.unsqueeze(0),
            16000,
            encoding="PCM_F",
            bits_per_sample=32,
            format="wav",
        )
    print(gen.size())
    # gen = gen.argmax(dim=-1)
    # gen = gen[:, gen_size:]

    # src_batch_tgt_mask = util.get_tgt_mask(src_batch.size(1))
    # src_batch_tgt_mask = src_batch_tgt_mask.to(device)
    # inf = model(src_batch)
    # inf = inf.argmax(dim=-1)


# inf = inf[0]
# print(inf.size())
# torchaudio.save(
#     "/scratch/other/sopi/CCREAIM/datasets/test/out/e2e_test_reconstruction.wav",
#     inf.cpu(),
#     16000,
#     encoding="PCM_F",
#     bits_per_sample=32,
#     format="wav",
# )
