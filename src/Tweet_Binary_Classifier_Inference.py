#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pickle
import requests
import os
# vect_url = 'https://drive.google.com/uc?export=download&id=1-BDg4YIu-CNHCwBUMhyXkEsun3M0Y3wQ'
# classf_url = 'https://drive.google.com/uc?export=download&id=1YsQO7LP5ezvDeh9MUlW6HRwKvWkXDYF6'
# if not os.exists('../models/tweet_vect.pkl'):
#   r = requests.get(vect_url, stream = True)
#   with open("../models/tweet_vect.pkl","wb") as f:
#     for chunk in r.iter_content(chunk_size=1024):
#       if chunk:
#         f.write(chunk)
# if not os.exists('../models/tweet_classf.pkl'):
#   r = requests.get(classf_url, stream = True)
#   with open("../models/tweet_classf.pkl","wb") as f:
#     for chunk in r.iter_content(chunk_size=1024):
#       if chunk:
#         f.write(chunk)

def mobile_tech_binary_classifier(inference_data):
    vectorizer = pickle.load(open('/content/AHSG-InterIIT/models/tweet_vect.pkl', 'rb'))
    classifier = pickle.load(open('/content/AHSG-InterIIT/models/tweet_classf.pkl', 'rb'))
    data = vectorizer.transform(inference_data['Text'])
    y_pred = classifier.predict(data)
    inference_data['Mobile_Tech'] = y_pred
    inference_data.loc[inference_data['num_brands']>0, 'Mobile_Tech'] = 1
    return inference_data


# In[ ]:


#import pandas as pd
#df = pd.read_pickle('C:/Users/SHIVAM/Downloads/article_df.pkl')
#mobile_tech_binary_classifier(df)

