import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.functional import accuracy


class ClassificationLitModule(pl.LightningModule):
    """
    Model-agnostic LightningModule for image classification.
    """

    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = accuracy(
            logits.softmax(dim=-1),
            y,
            task="multiclass",
            num_classes=self.hparams["model"]["num_classes"],
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = accuracy(
            logits.softmax(dim=-1),
            y,
            task="multiclass",
            num_classes=self.hparams["model"]["num_classes"],
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        opt_cfg = self.hparams["optimizer"]
        return torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
