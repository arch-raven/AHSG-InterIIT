import numpy as np
import pandas as pd 

import transformers
import torch
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