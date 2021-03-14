import numpy as np
import pandas as pd 

import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
import pytorch_lightning as pl

from argparse import ArgumentParser

class BinaryClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        main_text,
        mobile_tech_label,
    ):
        self.hparams = args
        self.main_text = main_text
        self.mobile_tech_label = mobile_tech_label
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_path)

    def __len__(self):
        return len(self.mobile_tech_label)

    def __getitem__(self, idx):
        # print("idx: ", idx)
        main_text = self.main_text[idx]
        mobile_tech_label = self.mobile_tech_label[idx]

        inputs = self.tokenizer(
            main_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.hparams.maxlen,
        )
        return {
            "ids_seq": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attn_masks": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "target": torch.tensor(mobile_tech_label, dtype=torch.float),
        }
        
class BinaryClassificationDataModule(pl.LightningDataModule):
    
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        
    def train_dataloader(self):
        article = pd.read_pickle("data/article_train_cleaned.pkl")
        tweet = pd.read_pickle("data/tweet_train_cleaned.pkl")
        
        article = article.loc[:,["Text", "Mobile_Tech_Flag"]]
        tweet = tweet.loc[:,["Tweet_with_emoji_desc", "Mobile_Tech_Tag"]].rename(columns={"Tweet_with_emoji_desc":"Text", "Mobile_Tech_Tag":"Mobile_Tech_Flag"})

        combined = pd.concat([article, tweet]).sample(frac=1.0)

        ds = BinaryClassificationDataset(self.hparams, combined.Text.to_list(), combined.Mobile_Tech_Flag.to_list())

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
    
    def val_dataloader(self):
        article = pd.read_pickle("data/article_dev_cleaned.pkl")
        tweet = pd.read_pickle("data/tweet_dev_cleaned.pkl")
        
        article = article.loc[:,["Text", "Mobile_Tech_Flag"]]
        tweet = tweet.loc[:,["Tweet_with_emoji_desc", "Mobile_Tech_Tag"]].rename(columns={"Tweet_with_emoji_desc":"Text", "Mobile_Tech_Tag":"Mobile_Tech_Flag"})

        combined = pd.concat([article, tweet])

        ds = BinaryClassificationDataset(self.hparams, combined.Text.to_list(), combined.Mobile_Tech_Flag.to_list())

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size*2,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

class DatasetForTokenClassification(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        texts,
        label_idxs,
    ):
        self.hparams = args
        self.texts = texts
        self.label_idxs = label_idxs
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # print("idx: ", idx)tokenize_and_align_labels
        assert type(idx) in [list, tuple, slice], f"idx should be an interable but instead received of type: {type(idx)}"
        texts = [self.texts[idxi] for idxi in idx]
        label_idxs = [self.label_idxs[idxi] for idxi in idx]

        return self.tokenize_and_align_labels(texts, label_idxs)
    
    def tokenize_and_align_labels(self, texts, label_idxs):
        tokenized_inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            is_split_into_words=False,
            max_length=self.hparams.maxlen,
            padding=True,
            return_tensors='pt',
        )

        labels = []
        for i, label in enumerate(label_idxs):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    if word_idx in label:
                        label_ids.append(1)
                    else:
                        label_ids.append(0)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return tokenized_inputs

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


class DataModuleForTokenClassification(pl.LightningDataModule):
    
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        
    def prepare_data(self):
        df = pd.read_pickle("data/dual_reviews_with_brandindexes.pkl")
        mask = df[['brand1','brand2']].isin(['ASUS', 'OnePlus'])
        mask = mask['brand1'] | mask['brand2']

        test_df = df.loc[mask]
        df = df.loc[~mask]

        val_df = df.sample(frac=0.1)
        train_df = df.loc[~df.index.isin(val_df.index.values),:]
        train_df = train_df.sample(frac=1)
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
    def train_dataloader(self):
        ds = DatasetForTokenClassification(self.hparams, self.train_df.review.to_list(), self.train_df.BrandIndexes.to_list())
        return SimpleBatchDataLoader(ds, shuffle=True, drop_last=True, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        ds = DatasetForTokenClassification(self.hparams, self.val_df.review.to_list(), self.val_df.BrandIndexes.to_list())
        return SimpleBatchDataLoader(ds, shuffle=False, drop_last=False, batch_size=self.hparams.batch_size)
    
    def test_dataloader(self):
        ds = DatasetForTokenClassification(self.hparams, self.test_df.review.to_list(), self.test_df.BrandIndexes.to_list())
        return SimpleBatchDataLoader(ds, shuffle=False, drop_last=False, batch_size=self.hparams.batch_size)



if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    #data related arguments
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    
    ##tokenizer and Language Model to use
    parser.add_argument("--base_path", type=str, default="xlm-roberta-base")
    
    args = parser.parse_args()
    
    dm = BinaryClassificationDataModule(args)
    dmT = DataModuleForTokenClassification(args)