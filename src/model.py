import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

import transformers
from transformers import AdamW, get_cosine_schedule_with_warmup


class MainModel(nn.Module):
    def __init__(self, args=None, **kwargs):
        super().__init__()
        self.base = transformers.AutoModel.from_pretrained(args.model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768,1)

    def forward(self, ids_seq, attn_masks, token_type_ids):
        base_out = self.base(
            ids_seq, attention_mask=attn_masks, token_type_ids=token_type_ids
        )
        # using maxpooled output
        max_out = self.dropout(base_out[1])
        return self.linear(max_out)


class SequenceClassicationLightningModule(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.save_hyperparameters(args)
        self.model = MainModel(self.hparams)
        
    @staticmethod
    def loss(logits, targets):
        return nn.BCEWithLogitsLoss()(logits, targets)

    def shared_step(self, batch):
        ids_seq, attn_masks, token_type_ids, target = (
            batch["ids_seq"],
            batch["attn_masks"],
            batch["token_type_ids"],
            batch["target"],
        )
        logits = self.model(ids_seq, attn_masks, token_type_ids)
        loss = self.loss(logits, target)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"valid_loss": loss, "logits": logits, "true_preds": batch["target"]}

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": self.model.base.parameters(), "lr": self.hparams.base_lr},
            {"params": self.model.linear.parameters(), "lr": self.hparams.linear_lr},
        ]
        optim = AdamW(grouped_parameters, lr=self.hparams.base_lr)

        # num_training_steps = (
        #     4863 // (self.hparams.batch_size * self.hparams.accumulate_grad_batches)
        # ) * self.hparams.max_epochs
        # sched = get_cosine_schedule_with_warmup(
        #     optim, num_warmup_steps=0, num_training_steps=num_training_steps
        # )

        # return [optim], [sched]
        return optim

    def validation_epoch_end(self, validation_step_outputs):
        y_pred = (
            torch.sigmoid(torch.cat([out["logits"] for out in validation_step_outputs]))
            .to("cpu")
            .detach()
            .numpy()
        )
        y_true = (
            torch.cat([out["true_preds"] for out in validation_step_outputs])
            .to("cpu")
            .detach()
            .numpy()
        )


if __name__ == "__main__":
    pass
