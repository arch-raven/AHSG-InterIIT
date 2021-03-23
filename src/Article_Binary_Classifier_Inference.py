# coding: utf-8

import pickle
import urllib
import os

# vect_url = 'https://drive.google.com/uc?export=download&id=174SJ90A4mBpPa1gQIylS63k4e5kE41Wt'
# classf_url = 'https://drive.google.com/uc?export=download&id=1LqlpZNgho3unOhAfjdai6GQInBkjX7VJ'
# if not os.path.exists('../models/article_vect.pkl'):
#   urllib.urlretrieve(vect_url, filename="../models/article_vect.pkl")
# if not os.path.exists('../models/article_classf.pkl'):
#   urllib.urlretrieve(classf_url, filename="../models/article_classf.pkl")
def mobile_tech_binary_classifier(inference_data):
    vectorizer = pickle.load(open('/content/AHSG-InterIIT/models/article_vect.pkl', 'rb'))
    classifier = pickle.load(open('/content/AHSG-InterIIT/models/article_classf.pkl', 'rb'))
    data = vectorizer.transform(inference_data['Text'])
    y_pred = classifier.predict(data)
    inference_data['Mobile_Tech'] = y_pred
    options = [['google'],['microsoft'],['asus'],['hp']]
    for option in options:
        indexes = inference_data.index[inference_data['brands'].apply(lambda x:x == option)]
        inference_data = inference_data.drop(indexes,axis = 0)
    inference_data.loc[inference_data['num_brands']>0, 'Mobile_Tech'] = 1
    return inference_data

#Sample Function Call - pass raw DataFrame only
#import pandas as pd
#df = pd.read_pickle('C:/Users/SHIVAM/Downloads/article_df.pkl')
#mobile_tech_binary_classifier(df)
