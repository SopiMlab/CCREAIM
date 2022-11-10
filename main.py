import logging
from pathlib import Path

import hydra
import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from utils import cfg_classes, dataset, util
from utils.cross_validation import cross_validation
from utils.test import test

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


class LogJobReturnCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: cfg_classes.BaseConfig, job_return: JobReturn, **kwargs
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")


@hydra.main(version_base=None, config_path="cfg", config_name="base")
def main(cfg: cfg_classes.BaseConfig):
    """The main entry point to the training loop/testing

    Args:
        cfg (DictConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    log.info(OmegaConf.to_yaml(cfg))

    util.set_seed(cfg.hyper.seed)

    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers
    if cfg.hyper.model != "transformer":
        data_root_sample_len = Path(cfg.data.data_root) / Path(
            "chopped_" + str(cfg.hyper.seq_len)
        )
        if not data_root_sample_len.exists():
            log.info(
                "Creating new chopped dataset with sample length: "
                + str(data_root_sample_len)
            )
            data_root_sample_len.mkdir()
            util.chop_dataset(
                cfg.data.original_data_root,
                str(data_root_sample_len),
                "mp3",
                cfg.hyper.seq_len,
            )
        # Sound dataset. Return name if testing
        data = dataset.AudioDataset(data_root_sample_len, cfg.hyper.seq_len)

    else:
        data_root = Path(cfg.data.data_root)
        if not data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data.data_root)
        # Feature dataset
        data = dataset.FeatureDataset(data_root)

    # Use gpu if available, move the model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Train/test
    if cfg.process.train:
        cross_validation(data, device, cfg)
    else:
        # Fetch the model:
        # testing load an existing trained one
        checkpoint = torch.load(cfg.logging.load_model_path, map_location="cpu")
        model = checkpoint["model"]
        model = model.to(device)

        # Make a dataloader
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=cfg.hyper.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.resources.num_workers,
        )
        test(model, dataloader, device, cfg)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=cfg_classes.BaseConfig)
    main()
