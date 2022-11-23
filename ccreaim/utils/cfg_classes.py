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
    save_one_per_batch: bool
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
    # number of cross val splits, 0= only training, 1= run testing on the training set, 1< number of splits
    cross_val_k: int


@dataclass
class Resources:
    # DataLoader num_workers parameter
    num_workers: int

    # General resource configs
    timeout_min: int
    cpus_per_task: Optional[int] = None
    gpus_per_node: Optional[int] = None
    tasks_per_node: int = 1
    mem_gb: Optional[int] = None
    nodes: int = 1

    # Slurm resource configs
    partition: Optional[str] = None
    qos: Optional[str] = None
    comment: Optional[str] = None
    constraint: Optional[str] = None
    exclude: Optional[str] = None
    gres: Optional[str] = None
    cpus_per_gpu: Optional[int] = None
    gpus_per_task: Optional[int] = None
    mem_per_gpu: Optional[int] = None
    mem_per_cpu: Optional[int] = None
    max_num_timeout: Optional[int] = None


@dataclass
class KldLossConfig:
    weight: float


@dataclass
class SpectralLossConfig:
    weight: float
    stft_bins: list[int]
    stft_hop_length: list[int]
    stft_window_size: list[int]


@dataclass
class ResAeConfig:
    levels: int
    downs_t: list[int]
    strides_t: list[int]
    input_emb_width: int
    block_width: int
    block_depth: int
    block_m_conv: float
    block_dilation_growth_rate: int
    block_dilation_cycle: int


@dataclass
class HyperConfig:
    model: str
    seed: int
    latent_dim: int
    seq_len: int
    # number of sequences for e2e_chunked, should be original audio length / seq_len
    num_seq: int
    # concatenation moe for e2e_chunked
    seq_cat: bool
    epochs: int
    batch_size: int
    learning_rate: float
    kld_loss: Optional[KldLossConfig] = None
    spectral_loss: Optional[SpectralLossConfig] = None
    res_ae: Optional[ResAeConfig] = None


@dataclass
class BaseConfig:
    logging: LoggingConfig
    process: ProcessConfig
    data: DataConfig
    resources: Resources
    hyper: HyperConfig
