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
    original_data_root: str
    data_root: str
    shuffle: bool


@dataclass
class Resources:
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
class HyperConfig:
    model: str
    seed: int
    latent_dim: int
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

    data: DataConfig
    hyper: HyperConfig
