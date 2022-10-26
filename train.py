import hydra
import torch
import wandb
from omegaconf import DictConfig

from model import ae, transformer, vae, vqvae
from utils import dataset, util


def train_step(model, batch):
    pass


def train(model, dataloader, cfg: DictConfig):
    for ep in range(cfg.epochs):
        train_step(model, ep)


@hydra.main(version_base=None, config_path="cfg", config_name="train")
def main(cfg: DictConfig):
    """The main entry point to the training loop

    Args:
        cfg (DictConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    if cfg.wandb:
        wandb.init(
            project="ccreaim",
            entity="ccreaim",
            name=f"{cfg.model}-{cfg.exp_name}-{str(cfg.seed)}-{str(cfg.run_id)}",
            group=f"{cfg.model}-{cfg.exp_name}",
            config=cfg,
        )

    util.set_seed(cfg.seed)

    if cfg.model == "ae":
        model = None
    elif cfg.model == "vae":
        model = None
    elif cfg.model == "vq-vae":
        model = None
    elif cfg.model == "transformer":
        model = None
    elif cfg.model == "end-to-end":
        model = None
    else:
        raise ValueError(f"Model type{cfg.model} is not defined!")

    if cfg.model != "transformer":
        # get sound dataset for training
        data = dataset.AudioDataset(cfg.data_path)
        print(len(data))
        dataloader = torch.utils.data.DataLoader(data)  # num_workers
        for da in dataloader:
            print(da.size())
            print(list(da.size())[1] / 16000)
            break
    else:
        # get feature dataset for training
        data = None
        dataloader = None

    train(model, dataloader, cfg)


if __name__ == "__main__":
    main()
