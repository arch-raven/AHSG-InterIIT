"""
Do this first (If you have linux run setup-utils.sh) else run the following manually:
============================ Requirements ==========================================
pip install spacy
pip install spacy-langdetect
python -m spacy download en_core_web_sm
pip install google_trans_new
pip install tqdm
pip install demoji
"""

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy_langdetect import LanguageDetector
from langdetect import DetectorFactory
DetectorFactory.seed = 5
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
from google_trans_new import google_translator
import time
import random 
from tqdm.auto import tqdm
import detect_script
import demoji
if demoji.last_downloaded_timestamp()==None:
  demoji.download_codes()
import re

def detect_lang(texts, truncate=True):
  """
  Input:
  texts: a list or a numpy array of strings
  Output:
  langs: the language of each text
  """
  langs = []
  for text in tqdm(texts):
      if truncate:
          text = text[:1000]
      processed = nlp(text)
      lang = processed._.language['language']
      prob = processed._.language['score']
      #print(lang,prob)
      if (lang!='en' and lang!='hi') or (lang=='en' and prob<0.9 and detect_script.detect(text)!='Devanagari'):
        lang = 'hing'
      elif (lang=='en' and detect_script.detect(text)=='Devanagari'):
        lang = 'hi'
      langs.append(lang)
  return langs

def _split_in_batches(article,max_len=5000):  
  """
  Implicit function to split text into batches of 5000 chars for TTS API
  """  
  batches = []
  while(len(article)>max_len):
    fstop_ind = article[:max_len].rfind('.')
    if(fstop_ind<0):
      fstop_ind = max_len-1
    batches.append(article[:fstop_ind+1])
    article = article[fstop_ind+1:]
  batches.append(article)
  return batches

def _translate_text(article, translator):
  translated_text = translator.translate(article,lang_src='hi',lang_tgt='en') 
  return translated_text



def translate(texts,hinglish=False):
    """
    !! This function will take considerable time ~ [1.5 sec X len(texts)] !!
    Input:
    texts: a list or numpy array of strings
    hinglish: boolean flag. set to True if you are translating hinglish (Google API)
              only supports 160 chars of hinglish
    Output:
    translated_texts: Translated to english
    num_trans: no of sentences translated successfully
    """
    text_batches = [] # List of lists
    for text in texts:
        if hinglish == True:
          batches = _split_in_batches(text,max_len=160)
        else:
          batches = _split_in_batches(text,max_len=5000)
        text_batches.append(batches)
    
    translated_texts = []
    try:
      for batches in tqdm(text_batches):
          translated_batches = []
          for batch in batches:
              interval = (((random.random()+1)*100)//1)/100
              time.sleep(interval) 
              translator = google_translator(timeout=5)
              translated_batch = _translate_text(batch, translator)
              translated_batches.append(translated_batch)
          translated_texts.append(' '.join(translated_batches))
    except:
        return translated_texts, len(translated_texts)
    return translated_texts, len(translated_texts)

def _clean_tweet(tweet, remove_emoji=True):
  tweet = tweet.replace('|',' ')
  tweet = tweet.replace('^','')
  tweet = tweet.replace('RT ','')
  tweet = tweet.replace('QT ','')
  tweet = re.sub(r"http\S+", "", tweet) 
  tweet = re.sub(r"\s+"," ",tweet)
  tweet = re.sub(r"@\w+[:]?", "", tweet)

  clean = re.sub(r"""
              [,.;:@#?!&$]+  # Accept one or more copies of punctuation
              \ *           # plus zero or more copies of a space,
              """,
              " ",          # and replace it with a single space
              tweet, flags=re.VERBOSE)
  if remove_emoji:
      tweet = demoji.replace(tweet)
  else:
      tweet = demoji.replace_with_desc(tweet)
  tweet = tweet.strip()
  return tweet

def clean_tweets(tweets):
  '''
  Takes a list of tweets and returns a list of cleaned tweets
  Input: 
  texts - a list of strings (tweet)
  Output:
  cleaned_tweets - a list of cleaned strings
  '''
  cleaned_tweets = []
  for tweet in tweets:
    cleaned_tweet = _clean_tweet(tweet)
    cleaned_tweets.append(cleaned_tweet)
  return cleaned_tweets


def _clean_article(article):
    article = article.replace('|',' ')
    article = article.replace('^','')
    article = re.sub(r"http\S+", "", article)
    article = re.sub(r"www.\S+", "", article)
    #article = re.sub(r'[.]\n?', ". ",article)
    article = re.sub(r'\s+', " ",article)
    article = article.strip()
    return article

def clean_articles(articles):
  cleaned_articles = []
  for article in articles:
    cleaned_article = _clean_article(article)
    cleaned_articles.append(cleaned_article)
  return cleaned_articles
