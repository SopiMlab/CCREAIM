import logging

import hydra
import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

from ccreaim.model import operate
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
        cfg (BaseConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    log.info(OmegaConf.to_yaml(cfg))

    util.set_seed(cfg.hyper.seed)

    # Use gpu if available, move the model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tmp_data_root = dataset.prepare_dataset_on_tmp(data_tar=cfg.data.data_tar)

    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers
    if "transformer" in cfg.hyper.model:
        # Feature dataset
        data = dataset.FeatureDataset(tmp_data_root)
    elif "bank-classifier" in cfg.hyper.model:
        # "Bank dataset"
        data = dataset.BankTransformerDataset(tmp_data_root)
    else:
        # Sound dataset
        data = dataset.AudioDataset(tmp_data_root, cfg.hyper.seq_len)

    # Train/test
    if cfg.process.train:
        cross_validation(data, device, cfg)
    else:
        # Fetch the model:
        # testing load an existing trained ones
        if (
            cfg.logging.load_model_path is None
            and cfg.hyper.pre_trained_ae_path is None
            and cfg.hyper.pre_trained_vqvae_path is None
            and cfg.hyper.pre_trained_transformer_path is None
        ):
            raise ValueError("No trained model path specified for testing.")

        if cfg.logging.load_model_path is not None:
            checkpoint = torch.load(cfg.logging.load_model_path, map_location="cpu")
            model_state_dict = checkpoint["model_state_dict"]
            hyper_cfg_schema = OmegaConf.structured(cfg_classes.HyperConfig)
            conf = OmegaConf.create(checkpoint["hyper_config"])
            cfg.hyper = OmegaConf.merge(hyper_cfg_schema, conf)
            log.info(
                f"Loading model with the following cfg.hyper:\n{OmegaConf.to_yaml(cfg.hyper)}"
            )
        get_model = operate.get_model_init_function(cfg.hyper)
        model = get_model()
        if cfg.logging.load_model_path is not None:
            model.load_state_dict(model_state_dict)
            log.info(f"Loaded model weights from {cfg.logging.load_model_path}")
        model = model.to(device)

        # Make a dataloader
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=cfg.hyper.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.resources.num_workers,
        )
        log.info(f"VALIDATION STARTED")
        test(model, dataloader, device, cfg)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=cfg_classes.BaseConfig)
    main()
