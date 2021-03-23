"""
Do this first (If you have linux run setup-utils.sh) else run the following manually:
============================ Requirements ==========================================
pip install spacy
pip install spacy-langdetect
python -m spacy download en_core_web_sm
pip install google_trans_new
pip install tqdm
pip install demoji
pip install syntok
pip install nltk
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
import syntok.segmenter as segmenter
import brands


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
  mentions = re.findall(r'\B@\w*[a-zA-Z]+\w*:*', tweet)
  for mention in mentions:
    if re.search(brands.search_exp,mention,re.IGNORECASE)!=None:
      brandname = re.findall(brands.search_exp,mention,re.IGNORECASE)[0]
      tweet = tweet.replace(mention,brandname)
    else:
      tweet = tweet.replace(mention,'')
  tweet = tweet.replace('|',' ')
  tweet = tweet.replace('^','')
  tweet = re.sub(r'\.{2}\.+\s*',' ', tweet, flags=re.VERBOSE)
  tweet = re.sub(r'RT\s+','', tweet)
  tweet = re.sub(r'QT\s+','', tweet)
  tweet = re.sub(r"http\S+", "", tweet) 
  tweet = re.sub(r"\s+"," ",tweet)
  tweet = re.sub(r"@\w+[:]?", "", tweet)
  if remove_emoji:
      tweet = demoji.replace(tweet)
  else:
      tweet = demoji.replace_with_desc(tweet)
  tweet = tweet.strip()
  return tweet

def clean_tweets(tweets, remove_emoji=True):
  '''
  Takes a list of tweets and returns a list of cleaned tweets
  Input: 
  texts - a list of strings (tweet)
  Output:
  cleaned_tweets - a list of cleaned strings
  '''
  cleaned_tweets = []
  for tweet in tweets:
    cleaned_tweet = _clean_tweet(tweet, remove_emoji)
    cleaned_tweets.append(cleaned_tweet)
  return cleaned_tweets


def _clean_article(article):
    article = re.sub(u'(\N{COPYRIGHT SIGN}|\N{TRADE MARK SIGN}|\N{REGISTERED SIGN})','', article)
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

def remove_space_before_dot(text):
  text = re.sub(r'(\w+)\s\.', r'\1.', text)
  text = re.sub(r'(\d+)\.\s(\d+)', r'\1.\2', text)
  return text

def split_into_sentences(text):
  text = remove_space_before_dot(text)
  sentences = []
  for paragraph in segmenter.process(text):
    for sentence in paragraph:
      sent = ' '.join([token.value for token in sentence])
      sent = remove_space_before_dot(sent)
      sentences.append(sent)
  return sentences

def decompose_by_rule(text):
    doc = nlp(text)
    idx = 0
    indices = []
    compound = []
    print("The subjects detected are:")
    for token in doc:
        compound.append(token.text)
        if token.dep == spacy.symbols.nsubj:
            print(token.text)
            indices.append(idx)
        idx += 1
    
    sentences = []
    cnt = 1
    start = 0
    #print("indices ",indices)
    #compound = text.split()
    while cnt < len(indices):
        sentences.append(compound[start:indices[cnt]])
        start = indices[cnt]
        cnt += 1
    sentences.append(compound[start:])
    return sentences

def segment_by_rule(text): ## Always ignore the first entry or key-value pair of the returned dictionary.
    #paragraph = paragraph.lower()
    brandslist = brands.get_brands([text],verbose=False)[0]
    list_of_sentences = split_into_sentences(text)
    brand_specific_chunks = dict()
    brand_specific_chunks["ignore"] = []
    for brandname in brandslist:
      brand_specific_chunks[brandname] = []
    #print(brand_specific_chunks)
    curr_brand = "ignore"
    for sentence in list_of_sentences:
        #tokens = sentence.split()
        brandlist_sent = brands.get_brands([sentence],verbose=False)[0]
        #print(brandlist_sent)
        if len(brandlist_sent) == 0:
          brand_specific_chunks[curr_brand].append(sentence)
        elif len(brandlist_sent) == 1:
          if brandlist_sent[0] in brand_specific_chunks.keys():
            curr_brand = brandlist_sent[0]
          brand_specific_chunks[curr_brand].append(sentence)
        elif len(brandlist_sent) > 1:
          #print('whoop!', sentence)
          brandlist_sent_sp = [r'\b'+w+r'\b' for w in brandlist_sent]
          b_patt = '|'.join(brandlist_sent_sp)
          #print(b_patt)
          curr_phrase = []
          for word in sentence.split(' '):
            #print(word)
            if len(re.findall(b_patt,word, re.IGNORECASE))>0 and (curr_brand != re.findall(b_patt,word, re.IGNORECASE)[0].lower()):
              if len(curr_phrase) > 0:
                brand_specific_chunks[curr_brand].append(' '.join(curr_phrase))
                curr_phrase = []
              #print('here')
              new_brandname = re.findall(b_patt,word, re.IGNORECASE)[0].lower()
              if new_brandname in brand_specific_chunks.keys():
                  curr_brand = re.findall(b_patt,word, re.IGNORECASE)[0].lower()
              #print('! ', curr_brand)
            curr_phrase.append(word)
          
          if len(curr_phrase)>0:
            brand_specific_chunks[curr_brand].append(' '.join(curr_phrase))
    remove_keys = set()
    for key in brand_specific_chunks.keys():
      if len(brand_specific_chunks[key]) == 0:
        remove_keys.add(key)
    remove_keys.add('ignore')
    brand_specific_chunks = {key:val for key, val in brand_specific_chunks.items() if key not in remove_keys}
    return brand_specific_chunks