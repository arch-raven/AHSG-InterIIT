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
        self.rouge = Rouge()
        if path is not None:
            self.model.load_state_dict(torch.load(path))
    def rouge_score(self,sentence,ref):
        s=''
        flag=0
        for char in sentence:
            if char=='<':
                flag=1
            elif char=='>':
                flag=0
            elif flag==0:
                s+=char

        gen=re.sub('\s+',' ',s).rstrip().lstrip()
        if gen=='':
            return 0,0,0
        rg=self.rouge.get_scores(gen,ref)
        r1,r2,rl=rg[0]["rouge-1"]['f'], \
        rg[0]["rouge-2"]['f'],rg[0]["rouge-l"]['f']

        return r1,r2,rl
    def generate_batch(self,data):
        output=random.sample(data,4)

        inp,label=[],[]
        for dat in output:
                inp.append(dat[0])
                label.append(dat[1])

        return inp,label
    def preprocess_article(self,articles):
        for article in articles:
            article = re.sub(r"http\S+", "", article)
            article = re.sub(r"www.\S+", "", article)
            article = re.sub(r"<\S+", "", article)
            article = re.sub('\n+', " ",article)
            article = article.strip()
        return articles
    
    def fit(self,articles, headlines):
        art = self.preprocess_article(articles)
        head = self.preprocess_article(headlines)
        assert len(art) == len(head)
        head=[]
        for i,j in zip(art,head):
            data.append([i,j])

    
        self.optimizer=optim.AdamW(model.parameters(),lr=0.00001)

        scalar=0
        val_score=0
        for i in range(3000):
                model.train()
                inp,label=self.generate_batch(data)
                input=self.tokenizer.prepare_seq2seq_batch(src_texts=inp, tgt_texts=label, padding=True, return_tensors='pt',max_length=600,truncation=True)
                outputs=self.model(input_ids=input['input_ids'].cuda(),labels=input['labels'].cuda())
                loss=outputs[0]

                scalar+=loss.item()
                torch.cuda.empty_cache()
                if(i+1)%50==0:
                        print('iteration={}, training loss={}'.format(i+1,scalar/(4*50)))
                        scalar=0

                loss.backward()
                self.optimizer.step()
    def predict(self,articles):
        articles = self.preprocess_article(articles)
        with torch.no_grad():
            preds=[]
            for line in articles:
                inp=[line]
                input=self.tokenizer.prepare_seq2seq_batch(src_texts=inp,
                                                      tgt_texts=label, padding=True, return_tensors='pt')

                output=self.model.generate(input_ids=input['input_ids'].cuda(),
                                      num_beams=5, early_stopping=True, max_length=20)
                out=self.tokenizer.batch_decode(output)
                torch.cuda.empty_cache()
                out[0] = re.sub(r"<\S+", "", out[0])
                preds.append(out[0])
            return preds
    def evaluate(self,actual,predicted):
        assert len(actual) == len(predicted), "No. of actual and predicted headlines shout be equal."
        act = self.preprocess_article(actual)
        pred = self.preprocess_article(predicted)
        with torch.no_grad():
            r1_,r2_,rl_=0,0,0
            candidate_corpus,references_corpus=[],[]

            for line in range(len(actual)):
                candidate_corpus.append(self.tokenizer.tokenize(pred[i]))
                references_corpus.append([self.tokenizer.tokenize(act[i])])

                r1,r2,rl= self.rouge_score(pred[i],act[i])
                    
                r1_+=r1
                r2_+=r2
                rl_+=rl

            r1_/=(len(actual)*0.01)
            r2_/=(len(actual)*0.01)
            rl_/=(len(actual)*0.01)
            bleu=0
            bleu=100*bleu_score(candidate_corpus, references_corpus)

            return {"R1": r1_, "R2": r2_, "RL": rl_, "BLEU": bleu}
