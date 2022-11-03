from dataclasses import dataclass


@dataclass
class LoggingConfig:
    save_logging: bool
    save_model: bool
    wandb: bool
    silent: bool
    exp_name: str
    run_suffix: int
    # Saved model destination
    model_root: str
    checkpoint: int
    # Output saving
    save_pred: bool
    pred_output: str
    save_encoder_output: bool
    encoder_output: str


@dataclass
class DataConfig:
    original_data_root: str
    data_root: str
    shuffle: bool


@dataclass
class SlurmConfig:
    gres: str


@dataclass
class HyperConfig:
    model: str
    seed: int
    seq_len: int
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass
class BaseConfig:
    # Logging
    logging: LoggingConfig

    # Train or test
    train: bool

    slurm: SlurmConfig
    data: DataConfig
    hyper: HyperConfig