import os
from glob import glob
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import transformers

from sentiment_classification import DatasetForTokenizedSentimentClassification, SimpleBatchDataLoader, TokenClassifier


class SentimentClassifier:
    def __init__(self, batch_size=16, for_inference=True, **kwargs):
        self.batch_size = batch_size
        gk_model = transformers.AutoModelForSequenceClassification.from_pretrained('ganeshkharad/gk-hinglish-sentiment', num_labels=3)
        self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.model = TokenClassifier(gk_model, for_inference)
        self.model.to(self.device)
        self.model.eval()
        del gk_model

    def predict(self, texts):

        ds = DatasetForTokenizedSentimentClassification(texts=texts)
        dl = SimpleBatchDataLoader(dataset=ds, shuffle=False, drop_last=False, batch_size=self.batch_size)

        ypreds = []
        brand_list = []
        
        for batch in dl:
            brand_list.extend(batch['brands'])
            batch.pop('brands')
            with torch.no_grad():
                outs = self.model(batch.to(self.device))
            ypreds.append(outs["logits"])

            
        y_pred = torch.softmax(torch.cat(ypreds), dim=-1).to("cpu").detach().numpy()
        y_pred = np.argmax(y_pred, axis=-1)
        print(y_pred, brand_list)
        results = [{}]*len(brand_list)
        for i in range(len(brand_list)):
            for br in brand_list[i]:
                results[i][br] = y_pred[i]
        
        return results

if __name__ == "__main__":
    pass
    
