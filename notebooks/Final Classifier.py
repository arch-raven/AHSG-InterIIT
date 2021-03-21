#!/usr/bin/env python
# coding: utf-8

# In[36]:


def mobile_tech_classifier(train_path, inference_path, text_identifier, tag_identifier):
    import time
    t0 = time.time()
    import pandas as pd
    import sklearn
    import pickle
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from xgboost import XGBClassifier
    
    df = pd.read_pickle(train_path)
    data_train = df[text_identifier]
    model = TfidfVectorizer()
    model.fit(data_train)   
    
    data_train = model.transform(df[text_identifier])
    y_train = df[tag_identifier]    
    
    xgbc = XGBClassifier(max_depth = 10,n_estimators = 100,random_state = 42,objective='binary:logitraw', n_jobs = -1, 
                     eval_metric = 'error')
    xgbc.fit(data_train,y_train)   
    
    df_test = pd.read_pickle(inference_path)
    data_test = df_test[text_identifier]
    y_test = df_test[tag_identifier]
    data_test = model.transform(data_test)
    y_pred = xgbc.predict(data_test)
    print("Accuracy: " + (str)(accuracy_score(y_test,y_pred)))
    print("Weighted f1: " + (str)(f1_score(y_test,y_pred,average = 'weighted')))
    print("Confusion Matrix:")
    print((str)(confusion_matrix(y_test,y_pred)))
    print(classification_report(y_test,y_pred))
    tf = time.time()
    print("Time taken: " + (str)(tf-t0) + " seconds")
    return y_pred


# In[37]:


mobile_tech_classifier('C:/Users/SHIVAM/Downloads/article_train_cleaned.pkl', 
                       'C:/Users/SHIVAM/Downloads/article_test_cleaned.pkl',
                       'Text', 'Mobile_Tech_Flag')


# In[38]:


mobile_tech_classifier('C:/Users/SHIVAM/Downloads/tweet_train_cleaned.pkl', 
                       'C:/Users/SHIVAM/Downloads/tweet_test_cleaned.pkl',
                       'Tweet', 'Mobile_Tech_Tag')

