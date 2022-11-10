import logging
from pathlib import Path

import torch
import torch.utils.data
import torchaudio
from omegaconf import OmegaConf

import wandb
from utils import cfg_classes, util

log = logging.getLogger(__name__)


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: cfg_classes.BaseConfig,
    fold: int = 0,
):

    if cfg.logging.wandb:
        wandb_group_name = f"{cfg.hyper.model}-{cfg.logging.exp_name}"
        wandb_exp_name = f"{cfg.hyper.model}-{cfg.logging.exp_name}-test-seed:{str(cfg.hyper.seed)}-id:{cfg.logging.run_id}"
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

    model.eval()
    with torch.no_grad():
        running_loss = torch.tensor(0.0)
        for batchnum, (seq, name) in enumerate(dataloader):
            seq = seq.to(device)
            loss, pred, _ = util.step(model, seq, device)

            if cfg.logging.save_pred:
                for p, n in zip(pred, name):
                    save_path = Path(cfg.logging.pred_output) / Path(n).name
                    if cfg.hyper.model == "transformer":
                        torch.save(p, save_path)
                    else:
                        torchaudio.save(  # type: ignore
                            save_path, p, 16000, encoding="PCM_F", bits_per_sample=32
                        )

            if cfg.logging.save_encoder_output:
                feat = model.encode(seq)  # type: ignore
                for f, n in zip(feat, name):
                    save_path = Path(cfg.logging.encoder_output) / Path(n).stem
                    torch.save(f.clone(), str(save_path) + ".pt")

            running_loss += loss.detach().cpu().item()

    if cfg.logging.wandb:
        wandb.log({"test/loss": running_loss})
        wandb.finish()

    if not cfg.logging.silent:
        log.info(f"Testing complete, total loss: {running_loss}")
