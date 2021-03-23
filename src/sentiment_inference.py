import os
from glob import glob
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn import metrics
import brands, utils
import torch
import transformers

from sentiment_classification import DatasetForTokenizedSentimentClassification, SimpleBatchDataLoader, TokenClassifier


class SentimentClassifier:
    def __init__(self, threshold=0.5, **kwargs):
        self.batch_size = batch_size
        gk_model = transformers.AutoModelForSequenceClassification.from_pretrained('ganeshkharad/gk-hinglish-sentiment', num_labels=3)
        self.device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.model = TokenClassifier(gk_model, threshold=threshold)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("ganeshkharad/gk-hinglish-sentiment")
        del gk_model

    def predict(self, list_of_dicts, is_tweets=True):
        if is_tweets:
            output_list_of_dicts = []
            for tweet_dict in list_of_dicts:
                output_list_of_dicts.append({'Text_ID':tweet_dict.pop('Text_ID')})
                brands_found = brands._get_brands(tweet_dict['Text'])       
                batch = self.tokenizer(
                        tweet_dict['Text'],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt',
                        is_split_into_words=False,
                    )
                with torch.no_grad():
                    outs = self.model(batch.to(self.device)
                
                for br in brands_found:
                    output_list_of_dicts[-1][br] = outs
            return output_list_of_dicts
            
        else:
            output_list_of_dicts = []
            for article_dict in list_of_dicts:
                output_list_of_dicts.append({'Text_ID':article_dict.pop('Text_ID')})
                
                for br, texts in article_dict.items():
                    batch = self.tokenizer(
                            " ".join(texts),
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt',
                            is_split_into_words=False,
                        )
                    with torch.no_grad():
                        outs = self.model(batch.to(self.device)
                    output_list_of_dicts[-1][br] = outs
            return output_list_of_dicts

if __name__ == "__main__":
    pass
    
