import logging

import hydra
import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from ccreaim.process.cross_validation import cross_validation
from ccreaim.process.test import test
from ccreaim.utils import cfg_classes, dataset, util

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

    # Use gpu if available, move the model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tmp_data_root = dataset.prepare_dataset_on_tmp(data_tar=cfg.data.data_tar, cfg=cfg)

    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers

    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers
    if cfg.hyper.model == "transformer":
        # Feature dataset
        data = dataset.FeatureDataset(tmp_data_root)
    elif cfg.hyper.model == "e2e-chunked":
        # Chunked sound dataset
        data = dataset.ChunkedAudioDataset(tmp_data_root, cfg.hyper.seq_len, seq_num=16)
    else:
        # Sound dataset
        data = dataset.AudioDataset(tmp_data_root, cfg.hyper.seq_len)

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
