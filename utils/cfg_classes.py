from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    wandb: bool
    silent: bool
    exp_name: str
    run_id: str
    # Saved model destination
    model_checkpoints: str
    load_model_path: str
    checkpoint: int
    # Output saving
    save_pred: bool
    pred_output: str
    save_encoder_output: bool
    encoder_output: str


@dataclass
class DataConfig:
    data_tar: str
    shuffle: bool


@dataclass
class ProcessConfig:
    # training or only testing
    train: bool
    # number of cross val splits
    # if 0=only training no validation, if 1=only one validation set etc.
    cross_val_k: int


@dataclass
class Resources:
    # DataLoader num_workers parameter
    num_workers: int

    # General resource configs
    timeout_min: int
    cpus_per_task: Optional[int]
    gpus_per_node: Optional[int]
    tasks_per_node: int
    mem_gb: Optional[int]
    nodes: int

    # Slurm resource configs
    partition: Optional[str]
    qos: Optional[str]
    comment: Optional[str]
    constraint: Optional[str]
    exclude: Optional[str]
    gres: Optional[str]
    cpus_per_gpu: Optional[int]
    gpus_per_task: Optional[int]
    mem_per_gpu: Optional[int]
    mem_per_cpu: Optional[int]
    max_num_timeout: Optional[int]


@dataclass
class SpectralLossConfig:
    weight: float
    stft_bins: list[int]
    stft_hop_length: list[int]
    stft_window_size: list[int]


@dataclass
class HyperConfig:
    model: str
    seed: int
    latent_dim: int
    seq_len: int
    epochs: int
    batch_size: int
    learning_rate: float
    spectral_loss: SpectralLossConfig


@dataclass
class BaseConfig:
    logging: LoggingConfig
    process: ProcessConfig
    data: DataConfig
    resources: Resources
    hyper: HyperConfig
