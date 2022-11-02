import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import wandb
from model import ae, transformer, vae, vqvae
from utils import dataset, util
from utils.test import test
from utils.train import train


@hydra.main(version_base=None, config_path="cfg", config_name="base")
def main(cfg: DictConfig):
    """The main entry point to the training loop/testing

    Args:
        cfg (DictConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    print(cfg)
    run_id = int(time.time())
    if cfg.wandb:
        wandb.init(
            project="ccreaim",
            entity="ccreaim",
            name=f"{cfg.model}-{cfg.exp_name}-{str(cfg.seed)}-{str(run_id)}",
            group=f"{cfg.model}-{cfg.exp_name}",
            config=cfg,
        )

    util.set_seed(cfg.seed)

    # Fetch the model:
    # if training initialize a new model, if testing load an existing trained one
    if cfg.train:
        if cfg.model == "ae":
            model = ae.get_autoencoder("base")
        elif cfg.model == "vae":
            model = None
        elif cfg.model == "vq-vae":
            model = None
        elif cfg.model == "transformer":
            model = transformer.get_transformer("base")
        elif cfg.model == "end-to-end":
            model = None
        else:
            raise ValueError(f"Model type {cfg.model} is not defined!")
    elif cfg.test:
        # Load an existing model if testing
        # TODO: instead of always loading _final.pth, load {model_name}_{cfg.some_name_for_training_suffix}?
        #       Or maybe too convoluted?
        exp_path, model_name = util.get_model_path(cfg)
        model = torch.load(exp_path / f"{model_name}_final.pth")
    else:
        raise ValueError("Didn't specify --config-name=train or --config-name=test")

    # Get the dataset, use audio data for any non-transformer model,
    # feature data for transformers
    if cfg.model != "transformer":
        data_root_sample_len = Path(cfg.data_root) / Path(
            "chopped_" + str(cfg.seq_length)
        )

        if not data_root_sample_len.exists():
            print(
                "Creating new chopped dataset with sample length: "
                + str(data_root_sample_len)
            )
            data_root_sample_len.mkdir()
            util.chop_dataset(
                cfg.original_data_root, str(data_root_sample_len), "mp3", cfg.seq_length
            )

        # Sound dataset. Return name if testing
        data = dataset.AudioDataset(
            data_root_sample_len, cfg.seq_length, return_name=cfg.test
        )

    else:
        data_root = Path(cfg.data_root) / Path("enc")  # + str(cfg.seq_length)

        if not data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data_root)
        # Feature dataset
        data = dataset.FeatureDataset(data_root)

    # Make a dataloader
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=2
    )

    # Use gpu if available, move the model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Train/test
    if cfg.train:
        optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
        train(model, dataloader, optimizer, device, cfg)
    elif cfg.test:
        test(model, dataloader, device, cfg)
    else:
        raise ValueError("Didn't specify --config-name=train or --config-name=test")


if __name__ == "__main__":
    main()
