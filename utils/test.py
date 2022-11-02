import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import wandb
from model import ae, transformer, vae, vqvae
from utils import dataset, util


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DictConfig,
):

    model.eval()
    with torch.no_grad():
        running_loss = torch.tensor(0.0)
        for batchnum, seq in enumerate(dataloader):
            seq, name = seq
            seq = seq.to(device)
            pred = model(seq)
            loss = model.loss_fn(pred, seq)

            if cfg.save_pred:
                for s, n in zip(seq, name):
                    save_path = Path(cfg.pred_output) / Path(n).name
                    torch.save(s, save_path)

            if cfg.save_encoder_output:
                feat = model.encode(seq)
                for f, n in zip(feat, name):
                    save_path = Path(cfg.encoder_output) / Path(n).stem
                    torch.save(f, str(save_path) + ".pth")

            running_loss += loss.detach().cpu().item()

            if not cfg.silent and batchnum % 100 == 0:
                print(f"{cfg.epochs:05d} - {batchnum}/{len(dataloader)} - loss: {loss}")
            if cfg.wandb:
                wandb.log({"loss": loss})

        if not cfg.silent:
            print(f"Epoch complete, total loss: {running_loss}")
