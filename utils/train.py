import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import wandb
from model import ae, transformer, vae, vqvae
from utils import dataset, util


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer,  # torch optimizer
    device: torch.device,
    cfg: DictConfig,
):

    exp_path, model_name = util.get_model_path(cfg)
    exp_path.mkdir(exist_ok=True)

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = torch.tensor(0.0)
        for batchnum, seq in enumerate(dataloader):
            seq = seq.to(device)
            if isinstance(model, transformer.Transformer):
                src = seq[:, :-1, :]
                tgt = seq[:, 1:, :]

                tgt_mask = model.get_tgt_mask(tgt.size(1))
                pred = model(src, tgt, tgt_mask)
                seq = tgt
            else:
                pred = model(seq)

            loss = model.loss_fn(pred, seq)
            running_loss += loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not cfg.silent and batchnum % 100 == 0:
                print(
                    f"{epoch + 1:03d}/{cfg.epochs:05d} - {batchnum}/{len(dataloader)} - loss: {loss}"
                )
            if cfg.wandb:
                wandb.log({"loss": loss})

        if not cfg.silent:
            print(f"Epoch complete, total loss: {running_loss}")

        if cfg.checkpoint != 0 and epoch % cfg.checkpoint == 0:
            save_path = exp_path / Path(f"{model_name}_ep-{epoch:03d}.pth")
            torch.save(model.cpu(), save_path)

    # Save final model
    final_save_path = exp_path / Path(f"{model_name}_final.pth")
    torch.save(model.cpu(), final_save_path)
