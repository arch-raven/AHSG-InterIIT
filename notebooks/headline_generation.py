"""
Headline Generator

"""
import os
import sys
import numpy as np 
import pandas as pd
import re
import random
import pickle
import torch
import torch.optim as optim
from torchtext.data.metrics import bleu_score
from rouge import Rouge
from transformers import T5ForConditionalGeneration, T5Tokenizer
class headline_gen():
    def __init__(self, path=None):
        
        self.model=T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        self.tokenizer=T5Tokenizer.from_pretrained('t5-base')
        if path is not None:
            self.model.load_state_dict(torch.load(path))
    def generate_batch(self,data):
        output=random.sample(data,4)

        inp,label=[],[]
        for dat in output:
                inp.append(dat[0])
                label.append(dat[1])

        return inp,label
    def preprocess_article(self,article):
        article = re.sub(r"http\S+", "", article)
        article = re.sub(r"www.\S+", "", article)
        article = re.sub(r"<\S+", "", article)
        article = re.sub('\n+', " ",article)
        article = article.strip()
        
        return article
    
    
    def fit(self,art, head):
        assert len(art) == len(head)
        data=[]
        for i,j in zip(art,head):
            data.append([i,j])
        for i in range(len(art)):
            data[i][0]= self.preprocess_article(data[i][0].lstrip().rstrip().lower())
            data[i][1]= self.preprocess_article(data[i][1].lstrip().rstrip().lower())
        

    
        self.optimizer=optim.AdamW(self.model.parameters(),lr=0.00001)

        scalar=0
        val_score=0
        for i in range(3000):
                self.model.train()
                inp,label=self.generate_batch(data)
                input=self.tokenizer.prepare_seq2seq_batch(src_texts=inp, tgt_texts=label, padding=True, return_tensors='pt',max_length=600,truncation=True)
                outputs=self.model(input_ids=input['input_ids'].cuda(),labels=input['labels'].cuda())
                loss=outputs[0]
                
                scalar+=loss.item()
                torch.cuda.empty_cache()
                del outputs
                del input
                del inp
                del label
                if(i+1)%50==0:
                        print('iteration={}, training loss={}'.format(i+1,scalar/(4*50)))
                        scalar=0

                loss.backward()
                self.optimizer.step()
    def predict(self,art):
        articles=[]
        for i in range(len(art)):
            articles.append(self.preprocess_article(art[i].lstrip().rstrip().lower()))
            
        
        with torch.no_grad():
            preds=[]
            for line in articles:
                inp=[line]
                input=self.tokenizer.prepare_seq2seq_batch(src_texts=inp, padding=True, return_tensors='pt')

                output=self.model.generate(input_ids=input['input_ids'].cuda(),
                                      num_beams=5, early_stopping=True, max_length=35)
                out=self.tokenizer.batch_decode(output)
                torch.cuda.empty_cache()
                out[0] = re.sub(r"<\S+", "", out[0])
                temp=out[0].split()
                if len(temp)>20:
                        temp=temp[:20]
                final=''.join(x+' ' for x in temp)
                final=final.lstrip().rstrip()
                
                preds.append(final)
             
            return preds
