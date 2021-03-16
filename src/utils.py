"""
Do this first (If you have linux run setup-utils.sh) else run the following manually:
============================ Requirements ==========================================
pip install spacy
pip install spacy-langdetect
python -m spacy download en_core_web_sm
pip install google_trans_new
"""

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy_langdetect import LanguageDetector
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
from google_trans_new import google_translator
import time
import random 

def detect_lang(texts, truncate=True):
  """
  Input:
  texts: a list or a numpy array of strings
  Output:
  langs: the language of each text
  """
  langs = []
  probs = []
  for text in texts:
      if truncate:
          text = text[:1000]
      processed = nlp(text)
      lang = processed._.language['language']
      if lang!='en':
          lang = 'hi'
      langs.append(lang)
  return langs

def _split_in_batches(article):  
  """
  Implicit function to split text into batches of 5000 chars for TTS API
  """  
  batches = []
  while(len(article)>5000):
    fstop_ind = article[:5000].rfind('.')
    if(fstop_ind<0):
      fstop_ind = 5000-1
    batches.append(article[:fstop_ind+1])
    article = article[fstop_ind+1:]
  batches.append(article)
  return batches

def _translate_text(article, translator):
  translated_text = translator.translate(article,lang_tgt='en') 
  return translated_text



def translate(texts):
    """
    !! This function will take considerable time ~ [1.5 sec X len(texts)] !!
    Input:
    texts: a list or numpy array of strings
    Output:
    translated_texts: Translated to english
    num_trans: no of sentences translated successfully
    """
    text_batches = [] # List of lists
    for text in texts:
        batches = _split_in_batches(text)
        text_batches.append(batches)
    
    translated_texts = []
    #try:
    for batches in text_batches:
        translated_batches = []
        for batch in batches:
            interval = (((random.random()+1)*100)//1)/100
            time.sleep(interval) 
            translator = google_translator(timeout=5)
            translated_batch = _translate_text(batch, translator)
            translated_batches.append(translated_batch)
        translated_texts.append(' '.join(translated_batches))

    # except:
    #     return translated_texts, len(translated_texts)
    return translated_texts, len(translated_texts)

    
