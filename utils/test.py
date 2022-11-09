import logging
from pathlib import Path

import torch
import torch.utils.data
import torchaudio

import wandb
from utils import cfg_classes, dataset, util

log = logging.getLogger(__file__)


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
):

    model.eval()
    with torch.no_grad():
        running_loss = torch.tensor(0.0)
        for batchnum, seq in enumerate(dataloader):
            seq, name = seq
            seq = seq.to(device)
            pred = model(seq)
            loss = model.loss_fn(pred, seq)

            if cfg.logging.save_pred:
                for p, n in zip(pred, name):
                    save_path = Path(cfg.logging.pred_output) / Path(n).name
                    if cfg.hyper.model == "transformer":
                        torch.save(p, save_path)
                    else:
                        torchaudio.save(
                            save_path, p, 16000, encoding="PCM_F", bits_per_sample=32
                        )

            if cfg.logging.save_encoder_output:
                feat = model.encode(seq)
                for f, n in zip(feat, name):
                    save_path = Path(cfg.logging.encoder_output) / Path(n).stem
                    torch.save(f, str(save_path) + ".pth")

            running_loss += loss.detach().cpu().item()

            if not cfg.logging.silent and batchnum % 100 == 0:
                log.info(
                    f"{cfg.hyper.epochs:05d} - {batchnum}/{len(dataloader)} - loss: {loss}"
                )
            if cfg.logging.wandb:
                wandb.log({"loss": loss})

        if not cfg.logging.silent:
            log.info(f"Epoch complete, total loss: {running_loss}")
