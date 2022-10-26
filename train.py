from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig

from model import ae, transformer, vae, vqvae
from utils import dataset, util


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.train()
    for batchnum, seq in enumerate(dataloader):
        seq.to(device)
        pred = model(seq)
        model.train_step(pred, seq)


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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if cfg.model == "ae":

        hidden_spec = {"input_ch": 16, "output_ch": 16, "kernel_size": 14, "stride": 1}

        enc_config = ae.EncoderConfig(
            INPUT_SIZE=1, LATENT_SIZE=200, HIDDEN_LAYER_SPECS=[hidden_spec]
        )
        dec_config = ae.DecoderConfig(LATENT_SIZE=200, HIDDEN_LAYER_SPECS=[hidden_spec])

        model = ae.AutoEncoder(enc_config, dec_config)

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

        # get sound dataset for training
        data = dataset.AudioDataset(data_root_sample_len)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1)  # num_workers

    else:
        if not cfg.data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data_root)
        # get feature dataset for training
        data = None
        dataloader = None

    model.to(device)
    train(model, dataloader, device)


if __name__ == "__main__":
    main()
