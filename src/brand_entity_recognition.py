import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataloader import DataModuleForTokenClassification
from model import LightningModuleForTokenClassification

class ToggleBaseTraining(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        
        params = list(pl_module.model.named_parameters())
        def is_backbone(n): return 'classifier' not in n
        
        if trainer.current_epoch == 0:
            print("-" * 100)
            print("ToggleBaseTraining Callback working.............")
            print(f"current_epoch is: {trainer.current_epoch} and freezing BASE layer's parameters")
            for n,p in params:
                if is_backbone(n): p.requires_grad = False
                else: p.requires_grad = True
            print("-" * 100)
        elif trainer.current_epoch == 1:
            print("-" * 100)
            print("ToggleBaseTraining Callback working.............")
            print(f"current_epoch is: {trainer.current_epoch} and unfreezing BASE layer's parameters for training")
            for n,p in params: p.requires_grad = True
            print("-" * 100)


class SaveModelWeights(pl.Callback):
    def __init__(self, save_from_epoch=1):
        self.save_from_epoch =save_from_epoch

    def on_validation_end(self, trainer, pl_module):
        os.makedirs("models/", exist_ok=True)
        print("-" * 100)
        print("SaveModelWeight Callback working.............")
        print(f"trainer.current_epoch: {trainer.current_epoch}")
        if trainer.current_epoch >= self.save_from_epoch:
            m_filepath = f"models/{pl_module.hparams.model_name}-epoch-{trainer.current_epoch}.pt"
            torch.save(pl_module.model.state_dict(), m_filepath)
            print(f"saved current model weights in file: {m_filepath}")
        print("-" * 100)

if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    # trainer related arguments
    parser.add_argument(
        "--gpus",
        default=1,
        help="if value is 0 cpu will be used, if string then that gpu device will be used",
    )
    parser.add_argument("--checkpoint_callback", action="store_true")
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--progress_bar_refresh_rate", default=0, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--model_name", default="ahsg", type=str)

    # data related arguments
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--maxlen", default=512, type=int)

    # model related arguments
    parser.add_argument("--base_path", type=str, default="xlm-roberta-base")
    parser.add_argument("--base_lr", default=1e-5, type=int)
    parser.add_argument("--linear_lr", default=5e-3, type=int)
    parser.add_argument("--base_dropout", default=0.3, type=float)
    parser.add_argument("--num_labels", default=1, type=int)
    parser.add_argument(
        "--bert_output_used",
        default="maxpooled",
        type=str,
        choices=["maxpooled", "weighted_sum"],
    )
    parser.add_argument("--run_name", default=None)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.effective_batch_size = args.batch_size * args.accumulate_grad_batches
    args.log_every_n_steps = args.accumulate_grad_batches * 5

    if not torch.cuda.is_available():
        args.gpus = 0

    pl_model = LightningModuleForTokenClassification(args)
    data = DataModuleForTokenClassification(args)

    if args.logger:
        args.logger = WandbLogger(
            project="ahsg", entity='professor',
            name=args.run_name if (args.run_name is not None) else None,
        )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            SaveModelWeights(save_from_epoch=0),
        ],
    )

    print(
        f"Training model_name={args.model_name} for epochs={args.max_epochs} with an effective_batch_size={args.effective_batch_size}"
    )
    trainer.fit(pl_model, data)
