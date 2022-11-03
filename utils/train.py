import logging
from pathlib import Path

import torch

import wandb
from model import ae, transformer, vae, vqvae
from utils import cfg_classes, dataset, util

log = logging.getLogger(__file__)


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer,  # torch optimizer
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
):

    exp_path, model_name = util.get_model_path(cfg)
    exp_path.mkdir(exist_ok=True)

    model.train()
    for epoch in range(cfg.hyper.epochs):
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

            if not cfg.logging.silent and batchnum % 100 == 0:
                log.info(
                    f"{epoch + 1:03d}/{cfg.hyper.epochs:05d} - {batchnum}/{len(dataloader)} - loss: {loss}"
                )
            if cfg.logging.wandb:
                wandb.log({"loss": loss})

        if not cfg.logging.silent:
            log.info(f"Epoch complete, total loss: {running_loss}")

        if (
            cfg.logging.checkpoint != 0
            and epoch % cfg.logging.checkpoint == 0
            and epoch != 0
        ):
            save_path = exp_path / Path(f"{model_name}_ep-{epoch:03d}.pth")
            torch.save(model.cpu(), save_path)

    # Save final model
    final_save_path = exp_path / Path(f"{model_name}_final.pth")
    torch.save(model.cpu(), final_save_path)
