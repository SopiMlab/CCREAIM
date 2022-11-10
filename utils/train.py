import logging
from pathlib import Path

import torch
import torch.utils.data

import wandb
from utils import cfg_classes, util

log = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer,  # torch optimizer
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
):

    checkpoints_path, model_name = util.get_model_path(cfg)
    checkpoints_path.mkdir(exist_ok=True)

    model.train()
    for epoch in range(1, cfg.hyper.epochs + 1):
        running_loss = torch.tensor(0.0)
        for batchnum, seq in enumerate(dataloader):
            seq = seq.to(device)
            loss, _, info = util.step(model, seq, device, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            if not cfg.logging.silent and batchnum % 100 == 0:
                log.info(
                    f"epoch: {epoch:03d}/{cfg.hyper.epochs:03d} - batch: {batchnum:05d}/{len(dataloader):05d} - loss: {loss}"
                )
            if cfg.logging.wandb:
                wandb.log({"loss": loss, "epoch": epoch, "batch": batchnum, **info})

        if not cfg.logging.silent:
            log.info(f"Epoch {epoch} complete, total loss: {running_loss}")

        if cfg.logging.checkpoint != 0 and epoch % cfg.logging.checkpoint == 0:
            save_path = checkpoints_path / Path(f"{model_name}_ep-{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss,
                },
                save_path,
            )

    # Save final model
    final_save_path = checkpoints_path / Path(f"{model_name}_final.pt")
    torch.save(
        {
            "epoch": cfg.hyper.epochs,
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0,
        },
        final_save_path,
    )
