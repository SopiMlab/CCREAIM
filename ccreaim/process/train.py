import logging
from pathlib import Path

import torch
import torch.utils.data
from omegaconf import OmegaConf

import wandb

from ..model import operate
from ..utils import cfg_classes, util

log = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
    fold: int = 0,
):
    optimizer = torch.optim.Adam(model.parameters(), cfg.hyper.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.hyper.lr_scheduler_gamma
    )
    if cfg.logging.wandb:
        wandb_group_name = f"{cfg.hyper.model}-{cfg.logging.exp_name}"
        wandb_exp_name = f"{cfg.hyper.model}-{cfg.logging.exp_name}-train-seed:{str(cfg.hyper.seed)}-id:{cfg.logging.run_id}"
        if fold != 0:
            wandb_exp_name += f"-fold:{fold}"

        wandb.init(
            project="ccreaim",
            entity="ccreaim",
            name=wandb_exp_name,
            group=wandb_group_name,
            config=OmegaConf.to_container(cfg),  # type: ignore
        )
        wandb.config.update({"time": cfg.logging.run_id})

    checkpoints_path, model_name = util.get_model_path(cfg)
    if cfg.process.cross_val_k > 1 and fold != 0:
        checkpoints_path /= Path(f"fold_{fold}")
    checkpoints_path.mkdir(exist_ok=True, parents=True)

    model.train()
    for epoch in range(1, cfg.hyper.epochs + 1):
        running_loss = torch.tensor(0.0)
        for batchnum, batch in enumerate(dataloader):
            loss, _, info = operate.step(model, batch, device, cfg.hyper)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            if not cfg.logging.silent and batchnum % 100 == 0:
                log.info(
                    f"epoch: {epoch:03d}/{cfg.hyper.epochs:03d} - batch: {batchnum:05d}/{len(dataloader):05d} - loss: {loss}"
                )
            if cfg.logging.wandb:
                wandb.log(
                    {"train/loss": loss, "epoch": epoch, "batch": batchnum, **info}
                )

        if not cfg.logging.silent:
            log.info(f"Epoch {epoch} complete, total loss: {running_loss}")

        if not cfg.logging.silent and cfg.hyper.lr_scheduler_gamma < 1.0:
            scheduler.step()
            log.info(f"Lowering learning rate to: {scheduler.get_last_lr()[-1]:.3e}")

        if cfg.logging.checkpoint != 0 and epoch % cfg.logging.checkpoint == 0:
            save_path = checkpoints_path / Path(f"{model_name}_ep-{epoch:03d}.pt")
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "hyper_config": OmegaConf.to_container(cfg.hyper),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": running_loss,
                    },
                    save_path,
                )
            except Exception as e:
                log.error(e)

    # Save final model
    final_save_path = checkpoints_path / Path(f"{model_name}_final.pt")
    try:
        torch.save(
            {
                "epoch": cfg.hyper.epochs,
                "model_state_dict": model.state_dict(),
                "hyper_config": OmegaConf.to_container(cfg.hyper),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": 0,
            },
            final_save_path,
        )
    except Exception as e:
        log.error(e)

    if cfg.logging.wandb:
        wandb.finish()
