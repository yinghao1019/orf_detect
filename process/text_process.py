from utils import load_special_tokens, set_log_config
import nltk
import os
import logging
import linecache
import re
import pickle
import argparse
import string
from collections import Counter
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
# #change dir to import customize module
import sys
sys.path.append(os.getcwd())

# customize module
logger = logging.getLogger(__name__)
en_stopwords = nltk.corpus.stopwords.words('english')

# define substitute regex pattern
link_p = re.compile(
    r'(https?://\S+|www\.\S+)|(\(?#*(EMAIL|PHONE|URL)_\w+#*\)?)')
emoji_p = re.compile("[" u"\U0001F600-\U0001F64F"  # emotions
                     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                     u"\U0001F680-\U0001F6FF"  # transport & map symbols
                     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                     u"\U00002702-\U000027B0"  # dinbats
                     u"\U000024C2-\U0001F521"
                     "]+", flags=re.UNICODE)
html_p = re.compile('<.*?>')
asc_p = re.compile(r'[^\w\.\-;\?!,\+\$/\>\<\[\]]+', flags=re.ASCII)
punc_p = re.compile(r'([_\*\=-]+)|(\.{2,})')
note_p = re.compile(r'\(.+?\)')
emphasize_p = re.compile(r'<(em|strong|b)>')
captial_p = re.compile(r"([a-z])([A-Z]\w)")
money_p = re.compile(r'\$\d+')
alpha_p = re.compile(r'\[?[a-zA-Z_]+\]?')

# define filter dirty text func


def replace_link(text):
    return link_p.sub('LINK ', str(text))


def remove_emoji(text):
    return emoji_p.sub(' ', str(text)).replace(':)', ' ')


def remove_htmlTag(text):
    return html_p.sub(' ', str(text))


def add_captial_space(text):
    return captial_p.sub(r"\1 \2", str(text))


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
        '&nbsp;', ' ').replace('&beta;', ' ').replace(u'\xa0', u' ')
    return text


def remove_note(text):
    return note_p.sub(' ', str(text))


def remove_punctuation(text):
    return punc_p.sub(' ', str(text))


def remove_nonASC(text):
    return asc_p.sub(' ', str(text))


def text_clean(text):
    """
    The clean pipeline contain replace link,remove emoji
    remove punc,split continous words,remove nonASCII

    Args:
     text(str):The text should filter noise

    Return:
     str:Text after clean
    """
    text = remove_emoji(replace_link(text))
    text = remove_htmlTag(remove_note(text))
    text = add_captial_space(remove_punctuation(replace_escChar(text)))
    text = remove_nonASC(text)
    return text.strip()


def has_money(text):
    result = money_p.search(text)
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
    return len(link_p.findall(str(text)))


def sent_process(sent):
    """
    This sents process contain tokenize,lemma converting,NER tag replace for input sent.

    Args:
     sent(spacy span object):The object is genrated for doc object

    Returns:
     list of str:After preprocess sent
    """
    # fetch sent start position
    sent_start = sent.start
    orig_sent = [t.lemma_.lower() for t in sent]

    # find all entity for sent and replace origin text phrase
    # #reversed to not modify the offsets of other entities when substituting
    for ent in reversed(sent.ents):
        orig_sent = orig_sent[:ent.start-sent_start] + \
            [f'[{ent.label_.lower()}]']+orig_sent[ent.end-sent_start:]

    return orig_sent


def remove_stopwords(sent):
    """The func will remove punc and non word string"""
    process_sent = []
    for w in sent:
        if (w not in string.punctuation) and (alpha_p.fullmatch(w) is not None):
            process_sent.append(w)

    return process_sent


def doc_process(doc):
    """
    This func used for  preproocess text pipeline.
    The pipeline will through tokenize,lemmatization,NER,remove stopwords

    Args:
     doc(spacy doc object): Generated by spacy language object.

    Returns:
     list of list str:The list of sents(list of token).
    """
    process_doc = []
    for sent in doc.sents:
        sent = sent_process(sent)  # lemmatize
        process_doc.append(remove_stopwords(sent))

    return process_doc


def corpus_process(data, nlp_pipe):
    """
    This will preprocess using text_clean,doc_process for corpus

    Args:
     data(list of str):the text data.
     nlp_pipe(spacy lang object): The pipeline object for text's lang.

    Returns:
     list of doc(list of list of doc):After preprocess text.
    """
    # remove text noise
    data = list(map(text_clean, data))
    # filter empty string
    data = filter(lambda x: True if x else False, data)
    # using nlp_pipeline to handle text
    data = list(map(lambda x: nlp_pipe(x.lower()), data))
    corpus = list(map(doc_process, data))
    logger.info('Process num:{}'.format(len(corpus)))

    return corpus
