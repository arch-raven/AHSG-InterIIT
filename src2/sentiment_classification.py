import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from sklearn import metrics

import transformers
from transformers import AdamW, get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

class SimpleBatchDataLoader:
    def __init__(self, dataset, shuffle=True, drop_last=False, batch_size=8):
        self.dataset = dataset
        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(self.sampler, drop_last=drop_last, batch_size=batch_size)

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for batch_idx in self.batch_sampler:
            yield self.dataset[batch_idx]


class DatasetForTextPairClassification(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        texts,
        brand_names,
        sentiments=None,
    ):
        self.hparams = args
        self.texts = texts
        self.brand_names = brand_names
        self.sentiments = sentiments if sentiments else [0]*len(texts)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # print("idx: ", idx)
        if type(idx)==int:
            texts = self.texts[idx]
            brand_names = self.brand_names[idx]
            sentiments = self.sentiments[idx]
        else:
            texts = [self.texts[idxi] for idxi in idx]
            brand_names = [self.brand_names[idxi] for idxi in idx]
            sentiments = [self.sentiments[idxi] for idxi in idx]

        tokenized_inps = self.tokenizer(
            brand_names,
            texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.maxlen,
            return_tensors='pt',
        )

        tokenized_inps['labels'] = torch.tensor(sentiments, dtype=torch.long)

        return tokenized_inps

class DataModuleForTextPairClassification(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.hparams = args

        self.filepath = args.filepath if args.filepath else "data/dual_product_reviews_formatted.csv"

    def prepare_data(self):
        df = pd.read_csv(self.filepath)
        mask = df["brand_names"].isin(['ASUS', 'OnePlus'])

        test_df = df.loc[mask]
        df = df.loc[~mask]

        val_df = df.sample(frac=0.1)
        train_df = df.loc[~df.index.isin(val_df.index.values),:]
        train_df = train_df.sample(frac=1)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def train_dataloader(self):
        ds = DatasetForTextPairClassification(self.hparams, self.train_df.texts.to_list(), self.train_df.brand_names.to_list(),self.train_df.sentiments.to_list())
        return SimpleBatchDataLoader(ds, shuffle=True, drop_last=True, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        ds1 = DatasetForTextPairClassification(self.hparams, self.val_df.texts.to_list(), self.val_df.brand_names.to_list(),self.val_df.sentiments.to_list())
        dl1 = SimpleBatchDataLoader(ds1, shuffle=False, drop_last=False, batch_size=self.hparams.batch_size*2)
        # ds2 = DatasetForTextPairClassification(self.hparams, self.train_df.texts.to_list(), self.train_df.brand_names.to_list(),self.train_df.sentiments.to_list())
        # dl2 = SimpleBatchDataLoader(ds2, shuffle=False, drop_last=False, batch_size=self.hparams.batch_size*2)
        return dl1

    def test_dataloader(self):
        ds = DatasetForTextPairClassification(self.hparams, self.test_df.texts.to_list(), self.test_df.brand_names.to_list(),self.test_df.sentiments.to_list())
        return SimpleBatchDataLoader(ds, shuffle=False, drop_last=False, batch_size=self.hparams.batch_size*2)
    

class LightningModuleForAutoModels(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.save_hyperparameters(args)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(self.hparams.base_path, num_labels=3)

    @staticmethod
    def loss_fn(logits, targets):
        ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.30,1.,0.10], device=logits.device))
        return ce(logits, targets)
    
    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        output = self.model(**batch)
        loss = self.loss_fn(output['logits'], labels)
        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "logits": output['logits'], "true_preds": labels}


    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        output = self.model(**batch)
        loss = self.loss_fn(output['logits'], labels)
        self.log(
            "valid_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return {"logits": output['logits'], "true_preds": labels}
    
    def test_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        output = self.model(**batch)
        loss = self.loss_fn(output['logits'], labels)
        self.log(
            "test_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return {"logits": output['logits'], "true_preds": labels}

    def configure_optimizers(self):
        params = list(self.model.named_parameters())
        
        def is_backbone(n): return 'classifier' not in n
        
        grouped_parameters = [
            {"params": [p for n,p in params if is_backbone(n)], "lr": self.hparams.base_lr},
            {"params": [p for n,p in params if not is_backbone(n)], "lr": self.hparams.linear_lr},
        ]
        optim = AdamW(grouped_parameters, lr=self.hparams.base_lr)
        
        # 55348 is the number of datapoints I am finetuning on, need to change it to some automatic method
        num_training_steps = (53348 // self.hparams.effective_batch_size) * self.hparams.max_epochs
        
        sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        return [optim], [sched]
#         return optim
    
    def training_epoch_end(self, training_step_outputs):
        
        y_pred = torch.cat([torch.argmax(out["logits"], dim=-1).view(-1) for out in training_step_outputs]).to("cpu").detach().numpy().reshape(-1)
        y_true = torch.cat([out["true_preds"].view(-1) for out in training_step_outputs]).to("cpu", dtype=int).detach().numpy().reshape(-1)
        
        mask = y_true != -100
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average="weighted")
        
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
    
    def validation_epoch_end(self, validation_step_outputs):

        y_pred = torch.cat([torch.argmax(out["logits"], dim=-1).view(-1) for out in validation_step_outputs]).to("cpu").detach().numpy().reshape(-1)
        y_true = torch.cat([out["true_preds"].view(-1) for out in validation_step_outputs]).to("cpu", dtype=int).detach().numpy().reshape(-1)

        mask = y_true != -100

        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average="weighted")
        print(f"validation epoch end on global_step: {self.global_step}\n", metrics.classification_report(y_true, y_pred))
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
    
    def test_epoch_end(self, test_step_outputs):

        y_pred = torch.cat([torch.argmax(out["logits"], dim=-1).view(-1) for out in test_step_outputs]).to("cpu").detach().numpy().reshape(-1)
        y_true = torch.cat([out["true_preds"].view(-1) for out in test_step_outputs]).to("cpu", dtype=int).detach().numpy().reshape(-1)

        mask = y_true != -100

        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average="weighted")
        
        print("test epoch end:\n", metrics.classification_report(y_true, y_pred))
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        
        
class SaveModelWeights(pl.Callback):
    def __init__(self, save_from_epoch=1):
        os.makedirs("models/", exist_ok=True)
        self.save_from_epoch =save_from_epoch
        self.call_num = 0

    def on_validation_end(self, trainer, pl_module):
        if (self.call_num+1)%3==0:
            print("-" * 100)
            print("SaveModelWeight Callback working.............")
            print(f"trainer.current_epoch: {trainer.current_epoch}")
            if trainer.current_epoch >= self.save_from_epoch:
                m_filepath = f"models/{pl_module.hparams.model_name}-epoch-{trainer.current_epoch}-{self.call_num}"
                while os.path.exists(m_filepath+".pt"):
                    m_filepath += "1"
                m_filepath += ".pt"
                torch.save(pl_module.model.state_dict(), m_filepath)
                print(f"saved current model weights in file: {m_filepath}")
            print("-" * 100)
        self.call_num += 1
        

if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    # trainer related arguments
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--checkpoint_callback", action="store_true")
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--progress_bar_refresh_rate", default=0, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--model_name", default="ahsg", type=str)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--val_check_interval", default=0.95, type=float)


    # data related arguments
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--maxlen", default=512, type=int)

    # model related arguments
    parser.add_argument("--base_path", type=str, default="xlm-roberta-base")
    parser.add_argument("--base_lr", default=1e-5, type=float)
    parser.add_argument("--linear_lr", default=5e-3, type=float)
    parser.add_argument("--num_labels", default=1, type=int)
    parser.add_argument("--bert_output_used", default="maxpooled", type=str,)
    parser.add_argument("--run_name", default=None)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.effective_batch_size = args.batch_size * args.accumulate_grad_batches
    args.log_every_n_steps = args.accumulate_grad_batches * 5

    if not torch.cuda.is_available():
        args.gpus = 0
    else: args.gpus = str(args.gpus)

    pl_model = LightningModuleForAutoModels(args)
    data = DataModuleForTextPairClassification(args)

    if args.logger:
        args.logger = WandbLogger(
            project="ahsg", entity='professor',
            name=args.run_name,
        )

    callbacks=[SaveModelWeights(save_from_epoch=0),]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    print(
        f"Training model_name={args.model_name} for epochs={args.max_epochs} with an effective_batch_size={args.effective_batch_size}"
    )
    trainer.fit(pl_model, data)
    trainer.test()
