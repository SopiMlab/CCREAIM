import logging
import random
import shutil
import tarfile
from pathlib import Path

import torch
import torch.utils.data
import torchaudio
from omegaconf import OmegaConf

import wandb
from ccreaim.model import ae

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

    if cfg.logging.save_encoder_output:
        encoder_output_length = ae.res_encoder_output_seq_length(cfg.hyper)
        tmp_encoder_output_tar_path = Path(
            f"/tmp/encodings_{encoder_output_length}_{cfg.logging.run_id}.tar"
        )
        log.info(f"Opening encodings output tar at: {str(tmp_encoder_output_tar_path)}")
        tmp_encoder_output_tar = tarfile.open(tmp_encoder_output_tar_path, "a")

    model.eval()
    with torch.inference_mode():
        running_loss = torch.tensor(0.0)
        for batchnum, batch in enumerate(dataloader):
            loss, pred, _ = operate.step(model, batch, device, cfg.hyper, batchnum)
            running_loss += loss.detach().cpu().item()

            if not cfg.logging.silent and batchnum % 100 == 0:
                log.info(f"batch: {batchnum:05d}/{len(dataloader):05d} - loss: {loss}")

            if cfg.logging.save_pred:
                save_root = Path(cfg.logging.pred_output)
                save_root.mkdir(exist_ok=True)
                n_predictions = 20
                if (
                    cfg.logging.save_one_per_batch
                    and batchnum % (len(dataloader) // n_predictions) == 0
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
                seq, _ = batch
                seq = seq.to(device)
                pred = model.encode(seq)  # type: ignore
                if isinstance(pred, tuple):
                    feat = pred[0]
                    inds = pred[1]
                    for f, i, n in zip(feat, inds, batch[1]):
                        f = f.clone().cpu()
                        i = i.clone().cpu()
                        util.save_to_tar(
                            tmp_encoder_output_tar,
                            {"feature": f, "embedding_indicies": i},
                            str(Path(n).stem) + ".pt",
                        )
                else:
                    feat = pred
                    for f, n in zip(feat, batch[1]):
                        f = f.clone().cpu()
                        util.save_to_tar(
                            tmp_encoder_output_tar,
                            {"feature": f},
                            str(Path(n).stem) + ".pt",
                        )

    if cfg.logging.save_encoder_output:
        log.info(f"Closing encodings output tar at: {str(tmp_encoder_output_tar_path)}")
        tmp_encoder_output_tar.close()
        log.info(
            f"Copying encoder output from {str(tmp_encoder_output_tar_path)} to {cfg.logging.encoder_output}"
        )
        shutil.move(
            tmp_encoder_output_tar_path,
            Path(cfg.logging.encoder_output) / tmp_encoder_output_tar_path.name,
        )

    if cfg.logging.wandb:
        wandb.log({"test/loss": running_loss})
        wandb.finish()

    if not cfg.logging.silent:
        log.info(f"Testing complete, total loss: {running_loss}")
