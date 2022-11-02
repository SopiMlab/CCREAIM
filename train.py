from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import wandb
from model import ae, transformer, vae, vqvae
from utils import dataset, util


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer,  # torch optimizer
    device: torch.device,
    cfg: DictConfig,
):

    exp_path = Path(cfg.model_path) / Path(cfg.exp_name)
    exp_path.mkdir(exist_ok=True)
    model_name = f"{cfg.model}_seqlen-{cfg.seq_length}_bs-{cfg.batch_size}_lr-{cfg.learning_rate}_seed-{cfg.seed}"

    model.train()
    for epoch in range(cfg.epochs):
        running_loss = torch.tensor(0.0)
        for batchnum, seq in enumerate(dataloader):
            seq = seq.to(device)
            pred = model(seq)
            loss = model.loss_fn(pred, seq)
            running_loss += loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not cfg.silent and batchnum % 100 == 0:
                print(
                    f"{epoch + 1:03d}/{cfg.epochs:05d} - {batchnum}/{len(dataloader)} - loss: {loss}"
                )
            if cfg.wandb:
                wandb.log({"loss": loss})

        if not cfg.silent:
            print(f"Epoch complete, total loss: {running_loss}")

        if cfg.checkpoint != 0 and epoch % cfg.checkpoint == 0:
            save_path = exp_path / Path(f"{model_name}_ep-{epoch:03d}.pth")
            torch.save(model.cpu().state_dict(), save_path)

    # Save final model
    final_save_path = exp_path / Path(f"{model_name}_final.pth")
    torch.save(model.cpu().state_dict(), final_save_path)


@hydra.main(version_base=None, config_path="cfg", config_name="train")
def main(cfg: DictConfig):
    """The main entry point to the training loop

    Args:
        cfg (DictConfig): The config object provided by Hydra

    Raises:
        ValueError: if misconfiguration
    """
    print(cfg)
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
        model = ae.get_autoencoder("base")
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
        data = dataset.AudioDataset(data_root_sample_len, cfg.seq_length)

    else:
        if not cfg.data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data_root)
        # get feature dataset for training
        data = None

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=2
    )  # num_workers
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    # move model to whatever device is goint to be used
    model = model.to(device)
    train(model, dataloader, optimizer, device, cfg)


if __name__ == "__main__":
    main()
