import logging

import torch
import torch.utils.data
from sklearn.model_selection import KFold

from ..model import operate
from ..utils.cfg_classes import BaseConfig
from . import test, train

log = logging.getLogger(__name__)


def cross_validation(
    dataset: torch.utils.data.Dataset, device: torch.device, cfg: BaseConfig
):
    get_model = operate.get_model_init_function(cfg.hyper)

    if cfg.process.cross_val_k > 1:
        kfold = KFold(
            cfg.process.cross_val_k,
            shuffle=cfg.data.shuffle,
            random_state=cfg.hyper.seed,
        )
        for fold, (tr_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
            model = get_model()
            model.to(device)
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
            train.train(model, train_dataloader, device, cfg, fold)

            log.info(f"FOLD {fold}/{cfg.process.cross_val_k} VALIDATION STARTED")
            test.test(model, test_loader, device, cfg, fold)

    elif cfg.process.cross_val_k == 1:
        model = get_model()
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.hyper.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.resources.num_workers,
        )
        log.info(f"TRAINING STARTED")
        train.train(model, dataloader, device, cfg)

        log.info(f"VALIDATION STARTED")
        test.test(model, dataloader, device, cfg)

    else:
        model = get_model()
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.hyper.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.resources.num_workers,
        )
        log.info(f"TRAINING STARTED")
        train.train(model, dataloader, device, cfg)
