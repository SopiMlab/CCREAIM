import logging
from pathlib import Path

import torch
import torch.utils.data

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
                for s, n in zip(seq, name):
                    save_path = Path(cfg.logging.pred_output) / Path(n).name
                    torch.save(s, save_path)

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
