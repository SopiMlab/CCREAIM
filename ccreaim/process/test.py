import logging
import random
from pathlib import Path

import torch
import torch.utils.data
import torchaudio
from omegaconf import OmegaConf

import wandb

from ..model import operate
from ..utils import util
from ..utils.cfg_classes import BaseConfig

log = logging.getLogger(__name__)


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: BaseConfig,
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
        for batchnum, batch in enumerate(dataloader):
            loss, pred, _ = operate.step(model, batch, device, cfg.hyper)

            if cfg.logging.save_pred:
                save_root = Path(cfg.logging.pred_output)
                save_root.mkdir(exist_ok=True)
                n_predictions = 20
                if (
                    cfg.logging.save_one_per_batch
                    and batchnum % (len(dataloader) / n_predictions) == 0
                ):
                    p, n = random.choice(list(zip(pred, batch[1])))
                    save_path = save_root / Path(n).name
                    util.save_model_prediction(
                        cfg.hyper.model, p.clone().cpu(), save_path
                    )
                elif not cfg.logging.save_one_per_batch:
                    for p, n in zip(pred, batch[1]):
                        save_path = save_root / Path(n).name
                        util.save_model_prediction(
                            cfg.hyper.model, p.clone().cpu(), save_path
                        )

            if cfg.logging.save_encoder_output:
                feat = model.encode(seq)  # type: ignore
                for f, n in zip(feat, batch[1]):
                    save_path = Path(cfg.logging.encoder_output) / Path(n).stem
                    torch.save(f.clone().cpu(), str(save_path) + ".pt")

            running_loss += loss.detach().cpu().item()

    if cfg.logging.wandb:
        wandb.log({"test/loss": running_loss})
        wandb.finish()

    if not cfg.logging.silent:
        log.info(f"Testing complete, total loss: {running_loss}")
