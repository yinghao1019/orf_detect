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
from nltk.corpus import wordnet, stopwords
from gensim.corpora.dictionary import Dictionary
# #change dir to import customize module
import sys
sys.path.append(os.getcwd())
#customize module
from utils import load_special_tokens,set_log_config,load_ftQuery
# add path module search path
logger = logging.getLogger(__name__)
# define substitute regex text
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
money_p = re.compile(r'\$\d+')


'''Match specific string func Region'''
def replace_link(text):
    return link_p.sub('LINK', str(text))

def remove_emoji(text):
    return emoji_p.sub(' ', str(text)).replace(':)', ' ')

def remove_htmlTag(text):
    return html_p.sub(' ', str(text))

def replace_escChar(text):
    """
    Remove any html escape character.
    Contain [&quot; &amp; &gt; &It; &nbsp; &beta;]


    Args:
        text(str):The text which you want to remove any html remove escape character
    
    Return:
        text(str):

    """
    text = text.replace('&quot;', ':').replace(
        '&amp;', 'and').replace('&It;', '<')
    text = text.replace('&gt;', '>').replace(
        '&nbsp;', ' ').replace('&beta;', ' ')
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
    text = remove_punctuation(replace_escChar(text))
    text = remove_nonASC(text)
    return text.strip()

def detect_link(text):
    # count emphaized html tag
    results = link_p.findall(text)
    if not results:
        return 0
    else:
        return 1
#find title whether contain Money symbol or not
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



def sent_process(sent):
  #fetch sent start position
  sent_start=sent.start
  orig_sent=[t.lemma_.lower() if t.text!='LINK' else t.text for t in sent]

  #find all entity for sent and replace origin text phrase 
  #reversed to not modify the offsets of other entities when substituting
  for ent in reversed(sent.ents):
    orig_sent=orig_sent[:ent.start-sent_start]+[ent.label_]+orig_sent[ent.end-sent_start:]
  return orig_sent
def remove_stopwords(sent):
  process_sent=[]
  for w in sent:
    if (w not in en_stopwords) and (w not in string.punctuation):
      process_sent.append(w)
  return process_sent
def doc_process(doc):
  process_doc=[]
  for sent in doc.sents:
    sent=sent_process(sent)#lemmatize
    process_doc.extend(remove_stopwords(sent))

  return process_doc


'''Vocab process funcs'''
def combine_data(data,col_names):
    # combined data non na data
    nonZero_df=data.loc[:,col_names].copy().notna().values.reshape(-1)
    df=data.loc[:,col_names].values.reshape(-1)
    return df[nonZero_df]

def corpus_process(data,col_names,nlp_pipe):
    #combine data
    data=combine_data(data,col_names).tolist()
    
    #remove noise
    data=list(map(text_clean,data))
    #using nlp_pipeline to handle text
    data=list(map(nlp_pipe,data))
    corpus=list(map(doc_process,data))

    return corpus


def build_vocab(corpus,spec_tokens,max_size,min_freq,speical_first):
    #count vocab
    vocab_count=Counter(corpus)
    #select top max_size word into vocab
    vocab=vocab_count.most_common(max_size)
    #filter word with freq smaller than vocab_freq
    vocab=[w for w,f in vocab if (f>min_freq) and (w not in spec_tokens)]
    #add special token
    if speical_first:
        vocab=spec_tokens+vocab
    else:
        vocab+=spec_tokens
    
    #convert to token2id format
    token2id={w:int(w_id) for w_id,w in enumerate(vocab)}            

    return vocab,token2id

def save_vocab_file(vocab,data_dir,file_path):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f'{data_dir} not exists!Create new one')

    try:
        with open(os.path.join(data_dir,file_path),'w',encoding='utf-8') as f_w:
            for word in vocab:
                f_w.write(str(word)+'\n')
        logger.info(f'Success save vocab file  to {data_dir}')
    except:
        logger.info('vocab File saved failed!')

def create_embeds(vocab,query,embed_path,args):
    #get pretrain vocab num and dim
    num,dim=map(int,linecache.getline(embed_path,1).strip().split(' '))
    #get vocab nums
    vocab_num=len(vocab.keys())
    #build storage vocab embed array
    vocab_embed=np.zeros((vocab_num,dim))

    logger.info(f'pretrain word embed nums:{num} dim:{dim}')
    logger.info(f'vocab nums:{vocab_num} \t top5 word pair:{list(vocab.items())[:5]}')
    logger.info(f'Embed vector shape:{vocab_embed.shape}')
    #get pretrain emebd_dim
    undefine_w=[]
    for w,w_id in vocab.items():
        if w in query:
            #get word data in pretrain file
            vector=linecache.getline(embed_path,int(query[w])+1).strip().split(' ')[1:]
            vocab_embed[w_id]=np.array(list(map(float,vector)))
        else:
            vocab_embed[w_id]=np.random.randn(dim)
            undefine_w.append(w)

    logger.info('Convert to fastText subword embed vector success!')
    logger.info('show pretrain embed:\n{}'.format(vocab_embed[20:23]))
    logger.info('Undefined word:{}'.format(len(undefine_w)))

    # except:
    #     logger.info('Loading pretrain embed error!')
    #     print('Embed_path:',embed_path)
            

    #save vector to disk
    save_dir=os.path.join(args.data_dir,args.task)
    #detect vocab dir
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(save_dir)
        logger.info(f'{save_dir} not exists!Create new one')

    try:
        #set path
        vocab_file_path=f'fastText_{dim}d_{vocab_num}_embed'

        #save!
        np.save(os.path.join(save_dir,vocab_file_path),vocab_embed)
        logger.info(f'Success save pretrain vector to {save_dir}')
    except:
        logger.info('save_vocab_failed!')




def main(args):

    #set path variable
    data_path=os.path.join(args.data_dir,args.task)
    embed_path=os.path.join(data_path,args.pretrain_embed_file)
    #load nlp pipeline
    spacy.require_gpu()
    en_nlp=spacy.load(args.model_type)

    # read data
    df = pd.read_csv(os.path.join(data_path,args.mode,'data.csv'),encoding=args.encode_format)
    special_tokens=load_special_tokens(args)
    pretrain_query=load_ftQuery(args)

    df = df.drop_duplicates(subset=['description','title'],keep=False)
    logger.info(f'Read data from {os.path.join(data_path,args.mode)} success!')
    logger.info(f'df shape:{df.shape}')
    logger.info(f'Special token: {special_tokens} nums: {len(special_tokens)}')

    #selected columns and build corpus with using nlp pipe
    #corpus shape = list of list of token
    # text_corpus=corpus_process(df,args.select_context_name,en_nlp)
    title_corpus=corpus_process(df,args.select_item_name,en_nlp)

    logger.info('****Start to build vocab****')
    # logger.info(f'Start to build context vocab for {args.select_context_name}!')
    # #build job context columns vocab
    # context_dict=Dictionary(text_corpus)
    # context_dict.filter_extremes(1,1,args.max_context_vocab)
    # #add special token
    # context_vocab=[w for w in set(context_dict.token2id).difference(set(special_tokens))]
    # if args.spec_first:
    #     context_vocab=special_tokens+context_vocab
    # else:
    #     context_vocab+=special_tokens
    # context_dict.token2id.update({w:w_id for w_id,w in enumerate(context_vocab)})

    # logger.info(f'vocab num : {len(context_vocab)}')
    # logger.info(f'top 15 vocab : {context_vocab[:15]}')
    # save_vocab_file(context_vocab,data_path,args.context_vocab_file)
    # context_dict.save(args.lda_vocab_file)

    logger.info(f'Start to build string vocab for {args.select_item_name}!')
    #flatten
    title_corpus=[w for doc in title_corpus for w in doc]

    #build title columns vocab
    title_vocab,title_id2token=build_vocab(title_corpus,special_tokens,args.max_title_vocab,
                                           args.min_title_freq,args.spec_first)
    logger.info('Create pretrain embed vector..')
    #create title embed vector
    create_embeds(title_id2token,pretrain_query,embed_path,args)

    '''Test Big file'''
    with open(os.path.join(data_path,args.context_vocab_file),'r',encoding='utf-8') as f_r:
        context_vocab=[w.strip() for w in f_r]
    #build dict
    context_token2id={str(w):i for i,w in enumerate(context_vocab)}
    create_embeds(context_token2id,pretrain_query,embed_path,args)
    #save_vocab
    # save_vocab_file(title_vocab,data_path,args.str_vocab_file)

    # logger.info(f'vocab size : {len(title_vocab)}')
    # logger.info(f'top 15 vocab : {title_vocab[:15]}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,help='',default='./Data')
    parser.add_argument('--task',type=str,default=None,required=True,choices=['fakeJob'],help='The training Model task.')
    parser.add_argument('--mode',type=str,default=None,required=True,choices=['train','test'],
                        help='Determined use train set or test set.')

    parser.add_argument('--str_vocab_file',type=str,default='str_vocab.txt',help='The file name for save string vocab')
    parser.add_argument('--context_vocab_file',type=str,default='context_vocab.txt',help='The file name for save context vocab.')
    parser.add_argument('--lda_vocab_file',type=str,default=r'saved_model\process_model\topic_model\lda_vocab.pkl',
                        help='The file name for lda vocab')
    parser.add_argument('--pretrain_embed_file',type=str,default=r'fastText\wiki-news-300d-1M-subword.vec',
                        help='The path for fastText embedding file.')

    parser.add_argument('--max_title_vocab',type=int,default=30000,help='The max vocab size for title.')
    parser.add_argument('--min_title_freq',type=int,default=3,help='The minimum frequency needed to include a token in title vocab.')
    parser.add_argument('--max_context_vocab',type=int,default=30000,help='Max vocab size for context word')
    parser.add_argument('--spec_first',action='store_true',
                        help='Whether to add special tokens into the vocabulary at first.If not call,special token will add into last')
    
    parser.add_argument('--select_context_name',type=str,nargs='+',choices=['company_profile','description','requirements','benefits'],
                       default=['company_profile','description','requirements','benefits'],help='Select data column for building context vocab.')
    parser.add_argument('--select_item_name',type=str,default='title',help='select data coulmn for build item vocab')

    parser.add_argument('--encode_format',type=str,default='utf-8',help="The read data's encoding format")
    parser.add_argument('--model_type',type=str,default='en_core_web_md',help='The model name for nlp process.')

    args=parser.parse_args()
    set_log_config()
    main(args)
