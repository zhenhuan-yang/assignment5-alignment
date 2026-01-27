from pathlib import Path
from omegaconf import OmegaConf

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class PathsConfig:
    train_bin: Path = MISSING
    valid_bin: Path = MISSING
    model_input: Path = MISSING
    model_output: Path = MISSING

    # NEW: deepspeed json path
    deepspeed_config: Path = MISSING


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    d_model: int = 768
    d_ff: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    rope_theta: float | None = 10000.0


@dataclass
class TrainingConfig:
    seed: int = 0
    dtype: str = "bfloat16"  # keep, but HF Trainer will use bf16 bool
    train_batch_size: int = 2
    eval_batch_size: int = "${training.train_batch_size}"

    train_steps: int = 100
    gradient_accumulation_steps: int = 16
    compile: bool = False
    eval_iterations: int = 50
    eval_interval: int = 500
    max_grad_norm: float | None = 1.0
    device: str = "cuda"
    lr: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    num_workers: int = 2
    pin_memory: bool = True
    attn_implementation: str = "flash_attention_2"

    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 10
    save_checkpoints: bool = True

    # -----------------------------
    # NEW: HuggingFace Trainer knobs
    # -----------------------------
    # use these in Trainer args
    max_steps: int = 2000
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # keep HF from dropping our columns
    remove_unused_columns: bool = False


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def register_configs():
    OmegaConf.register_new_resolver("eval", eval)
    cs = ConfigStore.instance()
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="paths", name="base_paths", node=PathsConfig)
    cs.store(name="base_config", node=Config)
