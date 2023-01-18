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

from ..model import ae, transformer, vqvae
from .cfg_classes import BaseConfig, HyperConfig, SpectralLossConfig

log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)


def chop_sample(sample: torch.Tensor, sample_length: int) -> list[torch.Tensor]:
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

            chopped_samples = chop_sample(full_sample.squeeze(), sample_length)
            for i, cs in enumerate(chopped_samples):
                out_name = str(pth.stem) + f"_{i:03d}" + ".wav"
                with io.BytesIO() as buffer:
                    try:
                        torchaudio.save(  # type: ignore
                            buffer,
                            cs.unsqueeze(0),
                            sample_rate,
                            encoding="PCM_U",
                            bits_per_sample=8,
                            format="wav",
                        )
                        buffer.seek(0)  # go to the beginning for reading the buffer
                        out_info = tarfile.TarInfo(name=out_name)
                        out_info.size = buffer.getbuffer().nbytes
                        out_tar.addfile(tarinfo=out_info, fileobj=buffer)
                    except Exception as e:
                        log.error(e)


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


# Calculates padding for nn.Conv1d-layers to achieve l_out=ceil(l_in/stride)
def conf_same_padding_calc(length: int, stride: int, kernel_size: int):
    out_length = math.ceil(float(length) / float(stride))

    if length % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (length % stride), 0)

    return math.ceil(pad / 2), out_length


# Calculates padding and output_padding for nn.ConvTranspose1d to get preferred length_out with minimal output_padding
def conf_same_padding_calc_t(
    length_in: int, length_out: int, stride: int, kernel_size: int
):
    padding = math.ceil(((length_in - 1) * stride - length_out + kernel_size) / 2)
    output_padding = length_out - ((length_in - 1) * stride - 2 * padding + kernel_size)
    return padding, output_padding


# Returns the path to the directory where a model is exported to/imported from according
# to configuration in cfg, as well as the base name of the model.
def get_model_path(cfg: BaseConfig):
    exp_path = Path(cfg.logging.model_checkpoints)
    model_name = f"{cfg.hyper.model}_seqlen-{cfg.hyper.seq_len}_bs-{cfg.hyper.batch_size}_lr-{cfg.hyper.learning_rate}_seed-{cfg.hyper.seed}"
    return exp_path, model_name


def load_pre_trained_ae(
    hyper_cfg: HyperConfig, encoder: ae.ResEncoder, decoder: ae.ResDecoder
) -> tuple[ae.ResEncoder, ae.ResDecoder]:
    checkpoint = torch.load(hyper_cfg.pre_trained_ae_path, map_location="cpu")
    pretrained_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

    if (
        hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
        and hyper_cfg.seq_len == pretrained_hyper_cfg.seq_len
        and hyper_cfg.res_ae.downs_t == pretrained_hyper_cfg.res_ae.downs_t
        and hyper_cfg.res_ae.strides_t == pretrained_hyper_cfg.res_ae.strides_t
        and hyper_cfg.res_ae.input_emb_width
        == pretrained_hyper_cfg.res_ae.input_emb_width
        and hyper_cfg.res_ae.block_width == pretrained_hyper_cfg.res_ae.block_width
        and hyper_cfg.res_ae.block_depth == pretrained_hyper_cfg.res_ae.block_depth
        and hyper_cfg.res_ae.block_m_conv == pretrained_hyper_cfg.res_ae.block_m_conv
        and hyper_cfg.res_ae.block_dilation_growth_rate
        == pretrained_hyper_cfg.res_ae.block_dilation_growth_rate
        and hyper_cfg.res_ae.block_dilation_cycle
        == pretrained_hyper_cfg.res_ae.block_dilation_cycle
    ):
        tmp_ae = ae.AutoEncoder(encoder, decoder)
        tmp_ae.load_state_dict(pretrained_state_dict)
        out_encoder = tmp_ae.encoder
        out_decoder = tmp_ae.decoder
        log.info(f"Loaded AE weights from {hyper_cfg.pre_trained_ae_path}")
        if hyper_cfg.freeze_pre_trained:
            out_encoder.requires_grad_(False)
            out_decoder.requires_grad_(False)
            log.info("Froze AE encoder and decoder weights.")
        return out_encoder, out_decoder
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "seq_len:\t\t\t\t"
            f"{hyper_cfg.seq_len}"
            "\t---\t"
            f"{pretrained_hyper_cfg.seq_len}\n"
            "res_ae.downs_t:\t\t\t\t"
            f"{hyper_cfg.res_ae.downs_t}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.downs_t}\n"
            "res_ae.strides_t:\t\t\t"
            f"{hyper_cfg.res_ae.strides_t}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.strides_t}\n"
            "res_ae.input_emb_width:\t\t\t"
            f"{hyper_cfg.res_ae.input_emb_width}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.input_emb_width}\n"
            "res_ae.block_width:\t\t\t"
            f"{hyper_cfg.res_ae.block_width}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_width}\n"
            "res_ae.block_depth:\t\t\t"
            f"{hyper_cfg.res_ae.block_depth}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_depth}\n"
            "res_ae.block_m_conv:\t\t\t"
            f"{hyper_cfg.res_ae.block_m_conv}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_m_conv}\n"
            "res_ae.block_dilation_growth_rate:\t"
            f"{hyper_cfg.res_ae.block_dilation_growth_rate}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_dilation_growth_rate}\n"
            "res_ae.block_dilation_cycle:\t\t"
            f"{hyper_cfg.res_ae.block_dilation_cycle}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_dilation_cycle}\n"
        )


def load_pre_trained_vqvae(
    hyper_cfg: HyperConfig,
    encoder: ae.ResEncoder,
    vq: vqvae.VectorQuantizer,
    decoder: ae.ResDecoder,
) -> tuple[ae.ResEncoder, vqvae.VectorQuantizer, ae.ResDecoder]:
    checkpoint = torch.load(hyper_cfg.pre_trained_vqvae_path, map_location="cpu")
    pretrained_state_dict = checkpoint["model_state_dict"]
    hyper_cfg_schema = OmegaConf.structured(HyperConfig)
    conf = OmegaConf.create(checkpoint["hyper_config"])
    pretrained_hyper_cfg = OmegaConf.merge(hyper_cfg_schema, conf)

    if (
        hyper_cfg.latent_dim == pretrained_hyper_cfg.latent_dim
        and hyper_cfg.seq_len == pretrained_hyper_cfg.seq_len
        and hyper_cfg.res_ae.downs_t == pretrained_hyper_cfg.res_ae.downs_t
        and hyper_cfg.res_ae.strides_t == pretrained_hyper_cfg.res_ae.strides_t
        and hyper_cfg.res_ae.input_emb_width
        == pretrained_hyper_cfg.res_ae.input_emb_width
        and hyper_cfg.res_ae.block_width == pretrained_hyper_cfg.res_ae.block_width
        and hyper_cfg.res_ae.block_depth == pretrained_hyper_cfg.res_ae.block_depth
        and hyper_cfg.res_ae.block_m_conv == pretrained_hyper_cfg.res_ae.block_m_conv
        and hyper_cfg.res_ae.block_dilation_growth_rate
        == pretrained_hyper_cfg.res_ae.block_dilation_growth_rate
        and hyper_cfg.res_ae.block_dilation_cycle
        == pretrained_hyper_cfg.res_ae.block_dilation_cycle
        and hyper_cfg.vqvae.num_embeddings == pretrained_hyper_cfg.vqvae.num_embeddings
    ):
        tmp_vq = vqvae.VQVAE(encoder, decoder, vq)
        tmp_vq.load_state_dict(pretrained_state_dict)
        out_encoder = tmp_vq.encoder
        out_vq = tmp_vq.vq
        out_decoder = tmp_vq.decoder
        log.info(f"Loaded VQ-VAE weights from {hyper_cfg.pre_trained_vqvae_path}")
        if hyper_cfg.freeze_pre_trained:
            out_encoder.requires_grad_(False)
            # vq is frozen in operate by emedding.grad = 0
            out_decoder.requires_grad_(False)
            log.info("Froze VQ-VAE encoder and decoder weights.")
        return out_encoder, out_vq, out_decoder
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "seq_len:\t\t\t\t"
            f"{hyper_cfg.seq_len}"
            "\t---\t"
            f"{pretrained_hyper_cfg.seq_len}\n"
            "res_ae.downs_t:\t\t\t\t"
            f"{hyper_cfg.res_ae.downs_t}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.downs_t}\n"
            "res_ae.strides_t:\t\t\t"
            f"{hyper_cfg.res_ae.strides_t}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.strides_t}\n"
            "res_ae.input_emb_width:\t\t\t"
            f"{hyper_cfg.res_ae.input_emb_width}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.input_emb_width}\n"
            "res_ae.block_width:\t\t\t"
            f"{hyper_cfg.res_ae.block_width}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_width}\n"
            "res_ae.block_depth:\t\t\t"
            f"{hyper_cfg.res_ae.block_depth}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_depth}\n"
            "res_ae.block_m_conv:\t\t\t"
            f"{hyper_cfg.res_ae.block_m_conv}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_m_conv}\n"
            "res_ae.block_dilation_growth_rate:\t"
            f"{hyper_cfg.res_ae.block_dilation_growth_rate}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_dilation_growth_rate}\n"
            "res_ae.block_dilation_cycle:\t\t"
            f"{hyper_cfg.res_ae.block_dilation_cycle}"
            "\t---\t"
            f"{pretrained_hyper_cfg.res_ae.block_dilation_cycle}\n"
            "vqvae.num_embeddings:\t\t\t"
            f"{hyper_cfg.vqvae.num_embeddings}"
            "\t---\t"
            f"{pretrained_hyper_cfg.vqvae.num_embeddings}\n"
        )


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
        and hyper_cfg.seq_len == pretrained_hyper_cfg.seq_len
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
        if hyper_cfg.freeze_pre_trained:
            trf.requires_grad_(False)
            log.info(f"Froze Transformer weights.")
        return trf
    else:
        raise ValueError(
            f"Pre-trained config is not matching current config:\n"
            "\t\t\t\t\tCurrent config\t---\tPre-trained config\n"
            "latent_dim:\t\t\t\t\t"
            f"{hyper_cfg.latent_dim}"
            "\t---\t"
            f"{pretrained_hyper_cfg.latent_dim}\n"
            "seq_len:\t\t\t\t\t"
            f"{hyper_cfg.seq_len}"
            "\t---\t"
            f"{pretrained_hyper_cfg.seq_len}\n"
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


# Spectral loss
class STFTValues:
    def __init__(self, n_bins: int, hop_length: int, window_size: int):
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.window_size = window_size


def norm(x: torch.Tensor):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def spec(seq: torch.Tensor, stft_val: STFTValues):
    return torch.norm(
        torch.stft(
            seq,
            stft_val.n_bins,
            stft_val.hop_length,
            win_length=stft_val.window_size,
            window=torch.hann_window(stft_val.window_size, device=seq.device),
        ),
        p=2,
        dim=-1,
    )


def spectral_loss(
    seq: torch.Tensor, pred: torch.Tensor, spectral_loss_cfg: SpectralLossConfig
) -> torch.Tensor:
    stft_val = STFTValues(
        spectral_loss_cfg.stft_bins[0],
        spectral_loss_cfg.stft_hop_length[0],
        spectral_loss_cfg.stft_window_size[0],
    )
    spec_in = spec(seq.float().squeeze(), stft_val)
    spec_out = spec(pred.float().squeeze(), stft_val)
    return norm(spec_in - spec_out)


def multispectral_loss(
    seq: torch.Tensor, pred: torch.Tensor, spectral_loss_cfg: SpectralLossConfig
) -> torch.Tensor:
    losses = torch.zeros(*seq.size()[:-1], device=seq.device)
    if losses.ndim == 1:
        losses = losses.unsqueeze(-1)
        seq = seq.unsqueeze(1)
        pred = pred.unsqueeze(1)
    args = (
        spectral_loss_cfg.stft_bins,
        spectral_loss_cfg.stft_hop_length,
        spectral_loss_cfg.stft_window_size,
    )
    for n_bins, hop_length, window_size in zip(*args):
        stft_val = STFTValues(n_bins, hop_length, window_size)
        for i in range(losses.size(-1)):
            spec_in = spec(seq[:, i], stft_val)
            spec_out = spec(pred[:, i], stft_val)
            losses[:, i] = norm(spec_in - spec_out)
    return losses


def spectral_convergence(
    seq: torch.Tensor,
    pred: torch.Tensor,
    spectral_loss_cfg: SpectralLossConfig,
    epsilon: float = 2e-3,
) -> torch.Tensor:
    stft_val = STFTValues(
        spectral_loss_cfg.stft_bins[0],
        spectral_loss_cfg.stft_hop_length[0],
        spectral_loss_cfg.stft_window_size[0],
    )
    spec_in = spec(seq.float().squeeze(), stft_val)
    spec_out = spec(pred.float().squeeze(), stft_val)
    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return (residual_norm * mask) / torch.clamp(gt_norm, min=epsilon)


def log_magnitude_loss(
    seq: torch.Tensor,
    pred: torch.Tensor,
    spectral_loss_cfg: SpectralLossConfig,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    stft_val = STFTValues(
        spectral_loss_cfg.stft_bins[0],
        spectral_loss_cfg.stft_hop_length[0],
        spectral_loss_cfg.stft_window_size[0],
    )
    spec_in = torch.log(spec(seq.float().squeeze(), stft_val) + epsilon)
    spec_out = torch.log(spec(pred.float().squeeze(), stft_val) + epsilon)
    return torch.abs(spec_in - spec_out)


def get_tgt_mask(size: int) -> torch.Tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
    return mask
