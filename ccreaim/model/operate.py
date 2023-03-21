import logging
from typing import Union

import torch
from torch.nn import functional as F

from ..utils import util
from ..utils.cfg_classes import HyperConfig
from . import decoder_only, transformer

log = logging.getLogger(__name__)


def step(
    model: torch.nn.Module,
    batch: Union[tuple[torch.Tensor, str], tuple[torch.Tensor, str, torch.Tensor]],
    device: torch.device,
    hyper_cfg: HyperConfig,
    batchnum: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}
    if "bank-classifier" in hyper_cfg.model:
        features, inds, _ = batch
        features = features.to(device)
        inds = inds.long().to(device)

        # Insert the zero-token, remove the last token
        tgt = torch.cat(
            (
                torch.zeros_like(features[:, 0:1, :], device=features.device),
                features[:, :-1, :],
            ),
            dim=1,
        )

        # Create the causal mask
        tgt_mask = util.get_tgt_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(device)
        pred = model(tgt, tgt_mask=tgt_mask)

        # Reshape the transformer output and indices for the loss function,
        # so that each correct index matches the corresponding logit output from
        # the transformer
        pred = pred.view(-1, hyper_cfg.transformer.vocab_size)
        inds = inds.view(-1)
        trf_auto = F.cross_entropy(pred, inds)

        loss = trf_auto
    else:
        raise ValueError(f"Model type {hyper_cfg.model} is not defined!")

    return loss, pred, info


def get_model_init_function(hyper_cfg: HyperConfig):
    # Model init function mapping
    if hyper_cfg.model == "bank-classifier":
        get_model = lambda: decoder_only.get_decoder(hyper_cfg)
    else:
        raise ValueError(f"Model type {hyper_cfg.model} is not defined!")
    return get_model
