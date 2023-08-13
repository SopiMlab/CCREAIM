from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    wandb: bool
    silent: bool
    exp_name: str
    run_id: str
    # Output saving
    save_pred: bool
    save_one_per_batch: bool
    pred_output: str
    # Saved model destination
    model_checkpoints: str
    checkpoint: int
    load_model_path: Optional[str] = None


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
class TransformerConfig:
    num_heads_latent_dimension_div: int
    num_enc_layers: int
    num_dec_layers: int
    dim_feedforward: int
    dropout: float
    autoregressive_loss_weight: float
    vocab_size: int
    linear_map: bool


@dataclass
class HyperConfig:
    model: str
    seed: int
    latent_dim: int
    seq_len: int  # basically context length, number of feature vectors. Unused but might come in handy
    num_seq: int
    seq_cat: bool
    epochs: int
    batch_size: int
    learning_rate: float
    lr_scheduler_gamma: float
    pre_trained_model_path: Optional[str] = None
    # These could be used to trigger util.load_pre_trained_transformer/decoder_only(),
    # but currently not implemented
    pre_trained_ae_path: Optional[str] = None
    pre_trained_vqvae_path: Optional[str] = None
    pre_trained_transformer_path: Optional[str] = None
    pre_trained_decoder_only_path: Optional[str] = None
    transformer: Optional[TransformerConfig] = None


@dataclass
class BaseConfig:
    logging: LoggingConfig
    process: ProcessConfig
    data: DataConfig
    resources: Resources
    hyper: HyperConfig


@dataclass
class LiveConfig:
    load_model_path: str
    input_device: str
    output_device: str
    sample_rate: int
    segment_length: int
    buffer_chunk_size: int
