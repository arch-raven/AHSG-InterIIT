import os
import pickle
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
    def __init__(self, bert_path,threshold=0.5, **kwargs):
        gk_model = transformers.AutoModelForSequenceClassification.from_pretrained('ganeshkharad/gk-hinglish-sentiment', num_labels=3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TokenClassifier(gk_model, threshold=threshold)
        self.model.load_state_dict(torch.load(bert_path))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("ganeshkharad/gk-hinglish-sentiment")
        del gk_model
        # with open(regression_path, 'rb') as fh:
        #     self.regression = pickle.load(fh)
        # with open(tfidf_path, 'rb') as fh:
        #     self.tfidf = pickle.load(fh)
        

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
                    outs = self.model(batch.to(self.device))
                #neg = self.predict_for_negative(tweet_dict['Text'])
                for br in brands_found:
                    #if neg==-1:
                  output_list_of_dicts[-1][br] = outs
                    #else:
                    #    output_list_of_dicts[-1][br] = 0
            return output_list_of_dicts
            
        else:
            output_list_of_dicts = []
            for article_dict in list_of_dicts:
                #print(article_dict)
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
                        outs = self.model(batch.to(self.device))
                    #neg = self.predict_for_negative(texts)
                    #if neg==-1:
                    output_list_of_dicts[-1][br] = outs
                    # else:
                    #     output_list_of_dicts[-1][br] = 0
            return output_list_of_dicts
        
    # def predict_for_negative(self, text):
    #     x = self.tfidf.transform(text)
    #     x = self.regression.predict_proba(x)
    #     if x>0.3: return 0
    #     else: return -1

if __name__ == "__main__":
    pass
    
