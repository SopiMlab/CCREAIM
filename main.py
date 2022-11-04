import logging
from pathlib import Path

import hydra
import submitit
import torch
import torch.utils.data
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import wandb
from model import ae, transformer, vae, vqvae
from utils import cfg_classes, dataset, util
from utils.test import test
from utils.train import train

log = logging.getLogger(__file__)


@hydra.main(version_base=None, config_path="cfg", config_name="base")
def main(cfg: cfg_classes.BaseConfig):
    """The main entry point to the training loop/testing

    Args:
        cfg (DictConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    env = submitit.JobEnvironment()

    log.info(OmegaConf.to_yaml(cfg))

    if cfg.logging.wandb:
        wandb.init(
            project="ccreaim",
            entity="ccreaim",
            name=f"{cfg.hyper.model}-{cfg.logging.exp_name}-{str(cfg.hyper.seed)}-{cfg.logging.run_id}",
            group=f"{cfg.hyper.model}-{cfg.logging.exp_name}",
            config=cfg,
        )

    util.set_seed(cfg.hyper.seed)

    # Fetch the model:
    # if training initialize a new model, if testing load an existing trained one
    if cfg.train:
        if cfg.hyper.model == "ae":
            model = ae.get_autoencoder("base", cfg.hyper.seq_len)
        elif cfg.hyper.model == "vae":
            model = vae.get_vae("base", cfg.hyper.seq_len)
        elif cfg.hyper.model == "vq-vae":
            model = None
        elif cfg.hyper.model == "transformer":
            model = transformer.get_transformer("base")
        elif cfg.hyper.model == "end-to-end":
            model = None
        else:
            raise ValueError(f"Model type {cfg.hyper.model} is not defined!")
    else:
        checkpoint = torch.load(cfg.logging.load_model_path)
        model = checkpoint["model"]

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
        data = dataset.AudioDataset(
            data_root_sample_len, cfg.hyper.seq_len, return_name=not cfg.train
        )

    else:
        data_root = Path(cfg.data.data_root) / Path("enc")  # + str(cfg.seq_length)
        if not data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data.data_root)
        # Feature dataset
        data = dataset.FeatureDataset(data_root)

    # Make a dataloader
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=cfg.hyper.batch_size, shuffle=cfg.data.shuffle, num_workers=2
    )

    # Use gpu if available, move the model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Train/test
    if cfg.train:
        optimizer = torch.optim.Adam(model.parameters(), cfg.hyper.learning_rate)
        train(model, dataloader, optimizer, device, cfg)
    else:
        test(model, dataloader, device, cfg)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=cfg_classes.BaseConfig)
    main()
