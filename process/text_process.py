import pandas as pd
import numpy as np
import nltk
import spacy
import os
import logging
import linecache
import re
import io
import pickle
import argparse
import string
from operator import itemgetter
from collections import Counter
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
# #change dir to import customize module
import sys
sys.path.append(os.getcwd())
#customize module
from utils import load_special_tokens,set_log_config,load_ftQuery


logger = logging.getLogger(__name__)

# define substitute regex pattern
en_stopwords=stopwords.words('english')
link_p = re.compile(r'(https?://\S+|www\.\S+)|(\(?#*(EMAIL|PHONE|URL)_\w+#*\)?)')
emoji_p = re.compile("[" u"\U0001F600-\U0001F64F"  # emotions
                     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                     u"\U0001F680-\U0001F6FF"  # transport & map symbols
                     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                     u"\U00002702-\U000027B0"  # dinbats
                     u"\U000024C2-\U0001F521"
                     "]+", flags=re.UNICODE)
html_p = re.compile('<.*?>')
asc_p = re.compile(r'[^\w\.\-;\?!,\+\$./\>\<\[\]]+', flags=re.ASCII)
punc_p = re.compile(r'([_\*\=-]+)|(\.{2,})')
note_p=re.compile(r'\(.+?\)')
emphasize_p = re.compile(r'<(em|strong|b)>')
captial_p=re.compile(r"([a-z])([A-Z])")
money_p = re.compile(r'\$\d+')
detect_link=re.compile('[LINK]')
#define rule based func
def replace_link(text):
    return link_p.sub('[LINK]', str(text))

def remove_emoji(text):
    return emoji_p.sub(' ', str(text)).replace(':)', ' ')

def remove_htmlTag(text):
    return html_p.sub(' ', str(text))
def add_captial_space(text):
    return captial_p.sub(r"\1 \2",str(text))
def replace_escChar(text):
    """
    Remove any html escape character.
    Contain [&quot; &amp; &gt; &It; &nbsp; &beta;]


    Args:
        text(str):The text which you want to remove any html remove escape character
    
    Return:
        text(str):

    """
    text = text.replace('&quot;', ': ').replace(
        '&amp;', 'and ').replace('&It;', '< ')
    text = text.replace('&gt;', '> ').replace(
        '&nbsp;', ' ').replace('&beta;', ' ').replace('\xa0',' ')
    return text
def remove_note(text):
    return note_p.sub(' ',str(text))

def remove_punctuation(text):
    return punc_p.sub(' ', str(text))

# 移除非ASCII 的字元
def remove_nonASC(text):
    return asc_p.sub(' ', str(text))

def text_clean(text):
    text = remove_emoji(replace_link(text))
    text = remove_htmlTag(remove_note(text))
    text = add_captial_space(remove_punctuation(replace_escChar(text)))
    text = remove_nonASC(text)
    return text.strip()


'''Feature engineering func'''
def has_money(text):
    result=money_p.search(text)
    if result is None:
        return 0
    else:
        return 1

def replace_locName(text):
    if text.startswith('GB'):
        text = 'United Kingdom'
    elif text.startswith('US'):
        text = 'United States'
    elif text.startswith('CA'):
        text = 'Canada'
    elif text.startswith('AU'):
        text = 'Australia'
    else:
        text = 'Other'

    return text

def count_links(text):
    '''Return LINK nums in text'''
    return len(detect_link.findall(str(text)))

'''Text process func'''
def sent_process(sent):
  #fetch sent start position
  sent_start=sent.start
  orig_sent=[t.lemma_.lower() for t in sent]

  #find all entity for sent and replace origin text phrase 
  #reversed to not modify the offsets of other entities when substituting
  for ent in reversed(sent.ents):
    orig_sent=orig_sent[:ent.start-sent_start]+[f'[{ent.label_.lower()}]']+orig_sent[ent.end-sent_start:]
  return orig_sent

def remove_stopwords(sent):
  process_sent=[]
  for w in sent:
    if (w not in string.punctuation) and (w.isalpha()):
      process_sent.append(w)
  return process_sent
def doc_process(doc):
  process_doc=[]
  for sent in doc.sents:
    sent=sent_process(sent)#lemmatize
    process_doc.append(remove_stopwords(sent))

  return process_doc

def corpus_process(data,nlp_pipe):
    #remove text noise
    data=list(map(text_clean,data))
    #filter empty string
    data=filter(lambda x:True if x else False,data)
    #using nlp_pipeline to handle text
    data=list(map(lambda x:nlp_pipe(x) ,data))
    corpus=list(map(doc_process,data))
    logger.info('Top 3 data {}'.format(corpus[:3]))

    return corpus
# rule_map={"campaignsrun":"campaigns run ","methodicallyMaintain":"methodically Maintain","skillsmotivated":"skills motivated","themYou":"them .You","businessAssisting":"business Assisting",
#           "CelebrationsLunchtime":"Celebrations Lunchtime","EnglishActive":"English Active","job Ceiling":"job Ceiling","GlanceA":"Glance A","ForBy":"For By",
# "platformWork":"platform Work","fiit":"fit","ManagerBasic":"Manager Basic","resourcesExcellent":"resources Excellent"," skillsSysadmin":" skills Sysadmin",
# "customersEstablish":"customers Establish","clientsyou":"clients you","projects Proficient":"projects proficient","routes An":"routes An","oilfied":"oil field"}