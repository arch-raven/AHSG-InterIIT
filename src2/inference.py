import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import pytorch_lightning as pl

from sentiment_classification import DatasetForTextPairClassification, SimpleBatchDataLoader, LightningModuleForAutoModels

def predict(args, dataframe, model, true_labels=False):

    ds = DatasetForTextPairClassification(args, dataframe.texts.to_list(), dataframe.brand_names.to_list(), dataframe.sentiments.to_list())
    dl = SimpleBatchDataLoader(dataset=ds, shuffle=False, drop_last=False, batch_size=args.batch_size)

    ypreds = []
    ytrue = []

    for batch in dl:
        # batch['labels'] =None
        with torch.no_grad():
            outs = model(**batch.to(args.device))
        ypreds.append(outs["logits"])
        ytrue.append(batch['labels'])

    y_pred = torch.softmax(torch.cat(ypreds), dim=-1).to("cpu").detach().numpy()
    y_true = torch.cat(ytrue).to("cpu", dtype=int).detach().numpy()

    df = pd.DataFrame(y_pred)
    df["y_true"] = y_true

    acc = metrics.accuracy_score(y_true, np.argmax(y_pred, axis=-1))
    f1 = metrics.f1_score(y_true, np.argmax(y_pred, axis=-1), average="weighted")
    print(f"----> accuracy_score: {acc},  f1_score: {f1}")
    print("----> classification report: \n", metrics.classification_report(y_true, np.argmax(y_pred, axis=-1)))
    return df


if __name__ == "__main__":
    pl.seed_everything(420)

    parser = ArgumentParser()

    parser.add_argument("path_to_ckpt", type=str, help="path to checkpoint file")

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

    # outputs prediction specific arguments
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv("data/sentiment_validation.tsv", sep='\t')
    test_df.rename(columns={"brand":"brand_names"}, inplace=True)

    pl_model = LightningModuleForAutoModels(args)
    pl_model.model.load_state_dict(torch.load(f"models/{args.path_to_ckpt}"))
    model = pl_model.model
    model.to(args.device)
    model.eval()

    outs = predict(args, test_df, model, true_labels=True)

    os.makedirs("outputs", exist_ok=True)
    outs.to_csv(f"outputs/{args.path_to_ckpt}.csv", index=False)
