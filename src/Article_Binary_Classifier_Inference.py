# coding: utf-8

import pickle
def mobile_tech_binary_classifier(inference_data):
    vectorizer = pickle.load(open('https://drive.google.com/file/d/174SJ90A4mBpPa1gQIylS63k4e5kE41Wt/view?usp=sharing', 'rb'))
    classifier = pickle.load(open('https://drive.google.com/file/d/1LqlpZNgho3unOhAfjdai6GQInBkjX7VJ/view?usp=sharing', 'rb'))
    data = vectorizer.transform(inference_data['Text'])
    y_pred = model.predict(data)
    inference_data['Mobile_Tech'] = y_pred
    return inference_data

#Sample Function Call - pass raw DataFrame only
#import pandas as pd
#df = pd.read_pickle('C:/Users/SHIVAM/Downloads/article_df.pkl')
#mobile_tech_binary_classifier(df)
