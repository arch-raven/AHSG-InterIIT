import numpy as np
import pandas as pd

import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
import pytorch_lightning as pl

from argparse import ArgumentParser


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


class DatasetForSeparateTextPairClassification(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        texts,
        brand_names,
        sentiments,
    ):
        self.hparams = args
        self.texts = texts
        self.brand_names = brand_names
        self.sentiments = sentiments

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_path)

    def __len__(self):
        return len(self.sentiments)

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

        texts_inp = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.hparams.maxlen,
            return_tensors='pt',
        )

        brand_names_inp = self.tokenizer(
            brand_names,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.hparams.maxlen,
            return_tensors='pt',
        )

        return {
            "texts":texts_inp,
            "brand_names":brand_names_inp,
            "labels":torch.tensor(sentiments, dtype=torch.long)
        }


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
        return SimpleBatchDataLoader(ds, shuffle=False, drop_last=False, batch_size=self.hparams.batch_size)



if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    #data related arguments
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)

    ##tokenizer and Language Model to use
    parser.add_argument("--base_path", type=str, default="xlm-roberta-base")

    args = parser.parse_args()

    dm = DataModuleForTextPairClassification(args)

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    text1, text2 = "Apple", "The new apple iphones received a lot of positive feedback from customers"
    tokenizer(text1,text2,padding=True, truncation=True, max_length=512)
