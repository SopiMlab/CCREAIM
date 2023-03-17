import io
import logging
import math
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from ..model import decoder_only, transformer
from .cfg_classes import BaseConfig, HyperConfig

log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)


def chop_sample(sample: torch.Tensor, sample_length: int) -> list[torch.Tensor]:
    if len(sample.size()) != 1:
        sample = torch.mean(sample, 0)
    assert len(sample.size()) == 1, "Sample is not 1 dimensional" + str(sample.size())
    chopped_samples_list: list[torch.Tensor] = []
    n_chops = len(sample) // sample_length
    for s in range(n_chops):
        chopped_samples_list.append(sample[s * sample_length : (s + 1) * sample_length])
    remainder = sample[n_chops * sample_length :]
    if remainder.size(0) > 0:
        chopped_samples_list.append(remainder)
    assert sum([len(chopped_sample) for chopped_sample in chopped_samples_list]) == len(
        sample
    ), f"Chopping did not maintain total sample length ({len(sample)})."
    return chopped_samples_list


def chop_dataset(in_root: str, out_tar_file_path: str, ext: str, sample_length: int):
    samples_paths = get_sample_path_list(Path(in_root), ext)
    with tarfile.open(out_tar_file_path, "a") as out_tar:
        for pth in samples_paths:
            try:
                full_sample, sample_rate = torchaudio.load(str(pth), format=ext)  # type: ignore
            except RuntimeError as e:
                log.warn(f"Could not open file, with error: {e}")
                continue

            try:
                chopped_samples = chop_sample(full_sample.squeeze(), sample_length)
                for i, cs in enumerate(chopped_samples):
                    out_name = str(pth.stem) + f"_{i:03d}" + ".wav"
                    with io.BytesIO() as buffer:
                        try:
                            torchaudio.save(  # type: ignore
                                buffer,
                                cs.unsqueeze(0),
                                sample_rate,
                                encoding="PCM_F",
                                bits_per_sample=32,
                                format="wav",
                            )
                            buffer.seek(0)  # go to the beginning for reading the buffer
                            out_info = tarfile.TarInfo(name=out_name)
                            out_info.size = buffer.getbuffer().nbytes
                            out_tar.addfile(tarinfo=out_info, fileobj=buffer)
                        except Exception as e:
                            log.error(e)
            except:
                print(f"Couldn't produce samples from path {pth}")


def save_to_tar(
    out_tar: tarfile.TarFile,
    data: dict[str, torch.Tensor],
    data_name: str,
):

    with io.BytesIO() as buffer:
        try:
            torch.save(data, buffer)
            buffer.seek(0)  # go to the beginning for reading the buffer
            out_info = tarfile.TarInfo(name=data_name)
            out_info.size = buffer.getbuffer().nbytes
            out_tar.addfile(tarinfo=out_info, fileobj=buffer)
        except Exception as e:
            log.error(e)


def save_model_prediction(model_name: str, pred: torch.Tensor, save_path: Path) -> None:
    try:
        if model_name == "transformer":
            torch.save(pred, save_path)
        elif "e2e-chunked" in model_name:
            torchaudio.save(  # type: ignore
                save_path,
                pred.flatten().unsqueeze(0),
                16000,
                encoding="PCM_F",
                bits_per_sample=32,
            )
        else:
            torchaudio.save(  # type: ignore
                save_path, pred, 16000, encoding="PCM_F", bits_per_sample=32
            )
    except Exception as e:
        log.error(e)


def get_sample_path_list(data_root: Path, ext: str = "mp3") -> list[Path]:
    return sorted(list(data_root.rglob(f"*.{ext}")))


# Returns the path to the directory where a model is exported to/imported from according
# to configuration in cfg, as well as the base name of the model.
def get_model_path(cfg: BaseConfig):
    exp_path = Path(cfg.logging.model_checkpoints)
    model_name = f"{cfg.hyper.model}_seqlen-{cfg.hyper.seq_len}_bs-{cfg.hyper.batch_size}_lr-{cfg.hyper.learning_rate}_seed-{cfg.hyper.seed}"
    return exp_path, model_name


def load_pre_trained_transformer(
    hyper_cfg: HyperConfig, trf: transformer.Transformer
) -> transformer.Transformer:
    checkpoint = torch.load(hyper_cfg.pre_trained_transformer_path, map_location="cpu")
    pretrained_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

    if (
        hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
        and hyper_cfg.vqvae.num_embeddings == pretrained_hyper_cfg.vqvae.num_embeddings
        and hyper_cfg.transformer.num_heads_latent_dimension_div
        == pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div
        and hyper_cfg.transformer.num_enc_layers
        == pretrained_hyper_cfg.transformer.num_enc_layers
        and hyper_cfg.transformer.num_dec_layers
        == pretrained_hyper_cfg.transformer.num_dec_layers
        and hyper_cfg.transformer.linear_map
        == pretrained_hyper_cfg.transformer.linear_map
    ):
        trf.load_state_dict(pretrained_state_dict)
        log.info(
            f"Loaded Transformer weights from {hyper_cfg.pre_trained_transformer_path}"
        )
        return trf
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "vqvae.num_embeddings:\t\t\t\t"
            f"{hyper_cfg.vqvae.num_embeddings}"
            "\t---\t"
            f"{pretrained_hyper_cfg.vqvae.num_embeddings}\n"
            "transformer.num_heads_latent_dimension_div: \t"
            f"{hyper_cfg.transformer.num_heads_latent_dimension_div}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div} \n"
            "transformer.num_enc_layers: \t\t\t"
            f"{hyper_cfg.transformer.num_enc_layers}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_enc_layers}\n"
            "transformer.num_dec_layers: \t\t\t"
            f"{hyper_cfg.transformer.num_dec_layers}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_dec_layers}\n"
            "transformer.linear_map:\t\t\t\t"
            f"{hyper_cfg.transformer.linear_map}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.linear_map}\n"
        )


def load_pre_trained_decoder_only(
    hyper_cfg: HyperConfig, trf: decoder_only.CachedDecoderOnly
) -> decoder_only.CachedDecoderOnly:
    checkpoint = torch.load(hyper_cfg.pre_trained_decoder_only_path, map_location="cpu")
    pretrained_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

    if (
        hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
        and hyper_cfg.vqvae.num_embeddings == pretrained_hyper_cfg.vqvae.num_embeddings
        and hyper_cfg.transformer.num_heads_latent_dimension_div
        == pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div
        and hyper_cfg.transformer.num_dec_layers
        == pretrained_hyper_cfg.transformer.num_dec_layers
        and hyper_cfg.transformer.linear_map
        == pretrained_hyper_cfg.transformer.linear_map
    ):
        trf.load_state_dict(pretrained_state_dict)
        log.info(
            f"Loaded Decoder-only weights from {hyper_cfg.pre_trained_transformer_path}"
        )
        return trf
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "vqvae.num_embeddings:\t\t\t\t"
            f"{hyper_cfg.vqvae.num_embeddings}"
            "\t---\t"
            f"{pretrained_hyper_cfg.vqvae.num_embeddings}\n"
            "transformer.num_heads_latent_dimension_div: \t"
            f"{hyper_cfg.transformer.num_heads_latent_dimension_div}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_heads_latent_dimension_div} \n"
            "transformer.num_dec_layers: \t\t\t"
            f"{hyper_cfg.transformer.num_dec_layers}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.num_dec_layers}\n"
            "transformer.linear_map:\t\t\t\t"
            f"{hyper_cfg.transformer.linear_map}"
            "\t---\t"
            f"{pretrained_hyper_cfg.transformer.linear_map}\n"
        )

def get_tgt_mask(size: int) -> torch.Tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
    return mask
