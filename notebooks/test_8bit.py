import sys

sys.path.append("..")
import torch
import torch.utils.data
import torchaudio
from omegaconf import OmegaConf
from torch.nn import functional as F

from ccreaim.model import operate
from ccreaim.utils import audio_tools, cfg_classes, dataset

data_tar = "/scratch/other/sopi/CCREAIM/datasets/test/out/8bit_test.tar"
load_transformer_path = "/scratch/other/sopi/CCREAIM/logs/2023-01-07/8bit-transformer_train_01-29-58/3/checkpoints/transformer_seqlen-1600_bs-32_lr-0.0001_seed-0_ep-004.pt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint = torch.load(load_transformer_path, map_location="cpu")
model_state_dict = checkpoint["model_state_dict"]
hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
conf = OmegaConf.create(checkpoint["hyper_config"])
hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)
get_model = operate.get_model_init_function(hyper_cfg)
model = get_model()
model.load_state_dict(model_state_dict)
model = model.to(device)


tmp_data_root = dataset.prepare_dataset_on_tmp(data_tar=data_tar)
data = dataset.Audio8BitDataset(tmp_data_root, hyper_cfg.seq_len)
dl = iter(torch.utils.data.DataLoader(data, batch_size=10, shuffle=False))

sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample, _ = next(dl)
sample_cat = sample.view(-1).unsqueeze(0)
print(sample.size())

torchaudio.save(
    "/scratch/other/sopi/CCREAIM/datasets/test/out/8bit_test_original.wav",
    sample_cat.cpu().to(torch.uint8),
    16000,
    encoding="PCM_U",
    bits_per_sample=8,
    format="wav",
)


with torch.inference_mode():
    seq = sample.to(device)
    src = F.one_hot(seq.long(), num_classes=256).int()
    src = src.to(device)
    gen = model.generate_chunks(src, 160, 160, 160)
    gen = gen.argmax(dim=-1)

gen_cat = gen.view(-1).unsqueeze(0)
print(gen_cat.size())
torchaudio.save(
    "/scratch/other/sopi/CCREAIM/datasets/test/out/8bit_test_generation.wav",
    gen_cat.cpu().to(torch.uint8),
    16000,
    encoding="PCM_U",
    bits_per_sample=8,
    format="wav",
)
