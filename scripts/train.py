"""
Train an MNIST classifier.

Usage
-----
    uv run scripts/train.py \
        --config configs/toy.yaml \
        --output-path outputs/toy
"""

import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from cenapad_mlpt.data import get_train_val_dataloaders
from cenapad_mlpt.models.utils import build_model_from_config
from cenapad_mlpt.training import ClassificationLitModule
from cenapad_mlpt.utils import (
    load_config,
    save_config,
    setup_reproducibility,
    silence_warnings_and_logs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MNIST classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(Path(args.config))
    setup_reproducibility(config["global"]["seed"])
    silence_warnings_and_logs()

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(out_dir / "config.yaml", config)

    train_loader, val_loader = get_train_val_dataloaders(config)

    model = build_model_from_config(config)
    lit_model = ClassificationLitModule(model, config)

    logger = TensorBoardLogger(save_dir=out_dir, name="tensorboard")
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        filename="best-val-acc-{epoch:03d}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        logger=logger,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        enable_model_summary=True,
    )

    trainer.fit(
        lit_model,
        train_loader,
        val_loader,
        ckpt_path=config["training"].get("resume_from", None),
    )


if __name__ == "__main__":
    main()
