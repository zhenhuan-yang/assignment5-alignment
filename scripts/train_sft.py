from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from rich.pretty import pprint as pprint
from rich.traceback import install

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

import wandb

from cs336_alignment.cs336_basics.train_config import Config, register_configs
from cs336_alignment.dataset import MemmapPackedDataset

register_configs()
logger = logging.getLogger(__name__)
install(show_locals=True)


@hydra.main(version_base=None, config_path=str(Path("configs").absolute().resolve()), config_name="config")
def main(cfg: Config) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    pprint(cfg_dict)

    # Resolve important paths
    train_path = str(cfg_dict["paths"]["train_bin"])
    valid_path = str(cfg_dict["paths"]["valid_bin"])
    model_input = str(cfg_dict["paths"]["model_input"])
    out_dir = str(cfg_dict["paths"]["model_output"])
    ds_cfg = str(cfg_dict["paths"]["deepspeed_config"])

    ctx_len = int(cfg_dict.get("model", {}).get("context_length", 512))

    tcfg = cfg_dict.get("training", {})
    seed = int(tcfg.get("seed", 42))
    lr = float(tcfg.get("lr", 2e-5))
    wd = float(tcfg.get("weight_decay", 0.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.03))

    per_device_train_bs = int(tcfg.get("per_device_train_batch_size", 1))
    per_device_eval_bs = int(tcfg.get("per_device_eval_batch_size", 1))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 16))

    max_steps = int(tcfg.get("max_steps", 2000))
    logging_steps = int(tcfg.get("logging_steps", 10))
    eval_steps = int(tcfg.get("eval_steps", 500))
    save_steps = int(tcfg.get("save_steps", 500))

    bf16 = bool(tcfg.get("bf16", True))
    grad_ckpt = bool(tcfg.get("gradient_checkpointing", True))
    attn_impl = str(tcfg.get("attn_implementation", "flash_attention_2"))

    wandb_project = tcfg.get("wandb_project", None)
    wandb_entity = tcfg.get("wandb_entity", None)

    # Tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(model_input, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe for packed fixed-length

    model = AutoModelForCausalLM.from_pretrained(
        model_input,
        torch_dtype=torch.bfloat16 if bf16 else None,
        attn_implementation=attn_impl,
    )

    # Required for gradient checkpointing memory savings
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Dataset (your packing behavior)
    train_ds = MemmapPackedDataset(train_path, seq_length=ctx_len)
    eval_ds = MemmapPackedDataset(valid_path, seq_length=ctx_len)

    # HF TrainingArguments + DeepSpeed
    report_to = ["wandb"] if wandb_project else ["none"]
    if wandb_project:
        # Optional: make HF Trainer + wandb happy
        wandb.init(project=wandb_project, entity=wandb_entity)

    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,

        max_steps=max_steps,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup_ratio,

        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs,
        gradient_accumulation_steps=grad_accum,

        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,

        bf16=bf16,
        fp16=False,

        report_to=report_to,
        run_name=Path(out_dir).name,

        deepspeed=ds_cfg,
        dataloader_num_workers=2,

        remove_unused_columns=False,  # keep input_ids/labels untouched
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")

    # Save final model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"Saved model+tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
