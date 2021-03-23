#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
def mobile_tech_binary_classifier(inference_data):
    vectorizer = pickle.load(open('https://drive.google.com/file/d/1-BDg4YIu-CNHCwBUMhyXkEsun3M0Y3wQ/view?usp=sharing', 'rb'))
    classifier = pickle.load(open('https://drive.google.com/file/d/1YsQO7LP5ezvDeh9MUlW6HRwKvWkXDYF6/view?usp=sharing', 'rb'))
    data = vectorizer.transform(inference_data['Tweet'])
    y_pred = model.predict(data)
    inference_data['Mobile_Tech'] = y_pred
    return inference_data


# In[ ]:


#import pandas as pd
#df = pd.read_pickle('C:/Users/SHIVAM/Downloads/article_df.pkl')
#mobile_tech_binary_classifier(df)

