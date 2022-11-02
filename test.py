import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import wandb
from model import ae, transformer, vae, vqvae
from utils import dataset, util


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DictConfig,
):

    model.eval()
    with torch.no_grad():
        running_loss = torch.tensor(0.0)
        for batchnum, seq in enumerate(dataloader):
            seq, name = seq
            seq = seq.to(device)
            pred = model(seq)
            loss = model.loss_fn(pred, seq)

            if cfg.save_pred:
                for s, n in zip(seq, name):
                    save_path = Path(cfg.pred_output) / Path(n).name
                    torch.save(s, save_path)

            if cfg.save_encoder_output:
                feat = model.encode(seq)
                for f, n in zip(feat, name):
                    save_path = Path(cfg.encoder_output) / Path(n).stem
                    torch.save(f, str(save_path) + ".pth")

            running_loss += loss.detach().cpu().item()

            if not cfg.silent and batchnum % 100 == 0:
                print(f"{cfg.epochs:05d} - {batchnum}/{len(dataloader)} - loss: {loss}")
            if cfg.wandb:
                wandb.log({"loss": loss})

        if not cfg.silent:
            print(f"Epoch complete, total loss: {running_loss}")


@hydra.main(version_base=None, config_path="cfg", config_name="test")
def main(cfg: DictConfig):
    """The main entry point to the training loop

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    exp_path = Path(cfg.model_path) / Path(cfg.exp_name)
    model_name = f"{cfg.model}_seqlen-{cfg.seq_length}_bs-{cfg.batch_size}_lr-{cfg.learning_rate}_seed-{cfg.seed}_final.pth"
    model = torch.load(exp_path / model_name)

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
        data = dataset.AudioDataset(
            data_root_sample_len, cfg.seq_length, return_name=True
        )

    else:
        if not cfg.data_root.exists():
            raise ValueError("Data folder does not exist: " + cfg.data_root)
        # get feature dataset for training
        data = None

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=2
    )  # num_workers

    # move model to whatever device is goint to be used
    model = model.to(device)
    test(model, dataloader, device, cfg)


if __name__ == "__main__":
    main()
