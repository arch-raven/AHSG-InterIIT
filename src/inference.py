import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import pytorch_lightning as pl

from model import MainModel
from dataloader import BinaryClassificationDataset

def predict(args, dataframe, model):
    
    ds = BinaryClassificationDataset(args, dataframe.Text.to_list(), dataframe.Mobile_Tech_Flag.to_list())
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    ypreds = []
    ytrue = []
    
    for batch in dl:
        ids_seq, attn_masks, target = (
            batch["ids_seq"].to(args.device),
            batch["attn_masks"].to(args.device),
            batch["target"],
        )
        logits = model(ids_seq, attn_masks).squeeze()
        ypreds.append(logits)
        ytrue.append(batch['target'])
    
    y_pred = torch.sigmoid(torch.cat(ypreds)).to("cpu").detach().numpy()
    y_true = torch.cat(ytrue).to("cpu", dtype=int).detach().numpy()
    
    return pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
        
    
if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    parser.add_argument("path_to_ckpt", type=str, help="path to checkpoint file")

    # data related arguments
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--maxlen", default=512, type=int)
    parser.add_argument("--base_path", type=str, default="xlm-roberta-base")
    
    # outputs prediction specific arguments
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MainModel(args)
    model.load_state_dict(torch.load(args.path_to_ckpt))
    model.to(args.device)
    model.eval()
    
    article = pd.read_pickle("data/article_dev_cleaned.pkl")
    tweet = pd.read_pickle("data/tweet_dev_cleaned.pkl")
    
    article = article.loc[:,["Text", "Mobile_Tech_Flag"]]
    tweet = tweet.loc[:,["Tweet_with_emoji_desc", "Mobile_Tech_Tag"]].rename(columns={"Tweet_with_emoji_desc":"Text", "Mobile_Tech_Tag":"Mobile_Tech_Flag"})

    tweet_outs = predict(args, tweet, model)
    article_outs = predict(args, article, model)
    
    os.makedirs("outputs", exist_ok=True)
    tweet_outs.to_csv(f"outputs/tweet_{args.path_to_ckpt}", index=False)
    article_outs.to_csv(f"outputs/article_{args.path_to_ckpt}", index=False)