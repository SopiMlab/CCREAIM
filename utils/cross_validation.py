import logging

import torch
import torch.utils.data
from sklearn.model_selection import KFold

import wandb
from model import ae, end_to_end, transformer, vae, vqvae
from utils import cfg_classes, test, train, util

log = logging.getLogger(__name__)


def cross_validation(
    dataset: torch.utils.data.Dataset, device: torch.device, cfg: cfg_classes.BaseConfig
):

    # Model init function mapping
    if cfg.hyper.model == "ae":
        get_model = lambda: ae.get_autoencoder(
            "base", cfg.hyper.seq_len, cfg.hyper.latent_dim
        )
    elif cfg.hyper.model == "vae":
        get_model = lambda: vae.get_vae("base", cfg.hyper.seq_len, cfg.hyper.latent_dim)
    elif cfg.hyper.model == "vq-vae":
        get_model = lambda: vqvae.get_vqvae(
            "base", cfg.hyper.seq_len, cfg.hyper.latent_dim
        )
    elif cfg.hyper.model == "transformer":
        get_model = lambda: transformer.get_transformer("base", cfg.hyper.latent_dim)
    elif cfg.hyper.model == "end-to-end":
        get_model = lambda: end_to_end.get_end_to_end(
            "base_ae", cfg.hyper.seq_len, 10, cfg.hyper.latent_dim
        )
    else:
        raise ValueError(f"Model type {cfg.hyper.model} is not defined!")

    if cfg.process.cross_val_k > 1:
        kfold = KFold(
            cfg.process.cross_val_k,
            shuffle=cfg.data.shuffle,
            random_state=cfg.hyper.seed,
        )
        for fold, (tr_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
            model = get_model()
            optimizer = torch.optim.Adam(model.parameters(), cfg.hyper.learning_rate)
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.hyper.batch_size,
                num_workers=cfg.resources.num_workers,
                sampler=torch.utils.data.SubsetRandomSampler(tr_idx),
            )
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.hyper.batch_size,
                num_workers=cfg.resources.num_workers,
                sampler=torch.utils.data.SubsetRandomSampler(val_idx),
            )
            log.info(f"FOLD {fold}/{cfg.process.cross_val_k} TRAINING STARTED")
            train.train(model, train_dataloader, optimizer, device, cfg, fold)

            log.info(f"FOLD {fold}/{cfg.process.cross_val_k} VALIDATION STARTED")
            test.test(model, test_loader, device, cfg, fold)
    else:
        model = get_model()
        optimizer = torch.optim.Adam(model.parameters(), cfg.hyper.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.hyper.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.resources.num_workers,
        )
        train.train(model, dataloader, optimizer, device, cfg)
