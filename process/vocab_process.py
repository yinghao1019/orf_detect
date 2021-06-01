import pandas as pd
import numpy as np
import spacy
import os
import logging
import argparse
import string
from collections import Counter
from gensim.models import keyedvectors,fasttext
from gensim.corpora.dictionary import Dictionary
import sys
sys.path.append(os.getcwd())# add path to module search path
#customize module
from utils import load_special_tokens,set_log_config
from text_process import corpus_process,en_stopwords,text_clean
logger = logging.getLogger(__name__)

'''Vocab process funcs'''
def combine_data(data,col_names):
    # combined data non na data
    nonZero_df=data.loc[:,col_names].copy().notna().values.reshape(-1)
    df=data.loc[:,col_names].values.reshape(-1)
    return df[nonZero_df]


def build_vocab(corpus,spec_tokens,max_size,min_freq,speical_first):
    """
    Used to create vocab dictionary(word index pair).
    In this process,set max vocab num & min_freq to filter word.
    also additionally add NER token to vocab.

    Args:
     corpus(list of text):The corpus for collect vocab.
     spec_tokens(list of token):Additional token that you want to add.
     max_size(int):The max vocab num.
     min_freq(int):The word in corpus must be higher than this freq
     special_first(bollean):Whether to add special token to first pos or not.

    Returns:
     list of str: collected vocab list
     dict: The token mapping index.
     
    """
    #count vocab
    vocab_count=Counter([w for text in corpus for w in text])
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

def create_embeds(vocab,model,embed_path,args):
    """
    Used for build pretrain embed vector.

    Args:
     vocab(dict):word to index mapping dict.
     model(gensim word2vec):The pretrain embed model
     embed_path:The dir for saving vocab embed vector.

    """
    #get vocab nums
    vocab_num=len(vocab.keys())
    #build storage vocab embed array
    vocab_embed=np.zeros((vocab_num,args.embed_dim))

    logger.info(f'Create word embed with fastText! \n nums:{vocab_num} dim:{args.embed_dim}')
    logger.info(f'vocab nums:{vocab_num} \t top5 word pair:{list(vocab.items())[:5]}')
    logger.info(f'Embed vector shape:{vocab_embed.shape}')
    #get pretrain emebd_dim
    item=0
    oov_words=[]
    for w,w_id in vocab.items():
        try:
            vocab_embed[w_id]=model[str(w)]
            item+=1
            if item%500==0:
                logger.info(f'Already convert {item} word to embed success!')
        except KeyError:
            logger.info(f'The {w_id} word {w} is not in pretrain model!')
            vocab_embed[w_id]=np.random.normal(0,0.1,args.embed_dim)
            oov_words.append(w)
    logger.info('show oov word of pretrain model: {}'.format(len(oov_words)))   
            

    #save vector to disk
    save_dir=os.path.join(embed_path,'vocab_embed')
    #detect vocab dir
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(save_dir)
        logger.info(f'{save_dir} not exists!Create new one')

    try:
        #set path
        vocab_file_path=f'{args.model_type}_{args.embed_type}_{args.embed_dim}d_{vocab_num}_embed'
        #save!
        np.save(os.path.join(save_dir,vocab_file_path),vocab_embed)
        logger.info(f'Success save pretrain vector to {save_dir}')
    except:
        logger.info('save_vocab_failed!')




def main(args):
    #set path variable
    data_path=os.path.join(args.data_dir,args.task)
    embed_path=f'{args.model_type}_{args.embed_type}_{args.corpora}_{args.embed_dim}d.kv'
    embed_path=os.path.join('./process/model/embed_model',embed_path)

    #load model for nlp pipeline & embed
    spacy.require_gpu()
    en_nlp=spacy.load(args.nlp_type)
    special_tokens=load_special_tokens(args)
    logger.info(f'loading pretrain embed model from {embed_path}')
    if os.path.isfile(embed_path):
        if args.model_type=='orig':
            kv_model=keyedvectors.KeyedVectors.load(embed_path)
        elif args.model_type=='ft':
            kv_model=fasttext.FastTextKeyedVectors.load(embed_path)
    else:
        raise FileNotFoundError('Embed file path incorrect!')

    # read data
    df = pd.read_csv(os.path.join(data_path,args.mode,'data.csv'),encoding=args.encode_format)
    df = df.drop_duplicates(subset=['description','title'],keep=False)

    logger.info(f'Read data from {os.path.join(data_path,args.mode)} success!')
    logger.info(f'df shape:{df.shape}')
    logger.info(f'Special token: {special_tokens} nums: {len(special_tokens)}')

    #build corpous using nlp pipeline
    logger.info('****Start to build vocab****')
    if args.select_context_name:
        logger.info(f'Start to build context vocab for {args.select_context_name}!')

        #text prepare
        context_data=combine_data(df,args.select_context_name).tolist()
        context_corpus=corpus_process(context_data,en_nlp)
        # build lda model vocab
        documents=[[w for sent in doc for w in sent] for doc in context_corpus]
        context_dict=Dictionary(documents)
        context_dict.filter_extremes(1,1,args.max_context_vocab)
        #add special token
        context_vocab=[w for w in set(context_dict.token2id).difference(set(special_tokens))]

        if args.spec_first:
            context_vocab=special_tokens+context_vocab
        else:
            context_vocab+=special_tokens
        context_dict.token2id.update({w:w_id for w_id,w in enumerate(context_vocab)})

        logger.info(f'context vocab num : {len(context_vocab)}')
        logger.info(f'top 15 context vocab : {context_vocab[:15]}')
        save_vocab_file(context_vocab,data_path,args.context_vocab_file)
        context_dict.save(args.lda_vocab_file)

        logger.info('Build pretrain embed model..')
        #build embed model
        logger.info('Create item_vocab pretrain embed!')
        create_embeds(context_dict.token2id,kv_model,data_path,args)

    if args.select_item_name:
        
        logger.info(f'Start to build string vocab for {args.select_item_name}!')

        #data prepare
        item_data=df[args.select_item_name].values.tolist()
        item_corpus=list(map(text_clean,item_data))#clean text

        #build title vocab 
        item_corpus=[[w.lower() for w in doc.split() if (w not in string.punctuation) and w.isalpha()]
                     for doc in item_corpus]
        item_vocab,item_token2id=build_vocab(item_corpus,special_tokens,args.max_item_size,
                                               args.min_item_freq,args.spec_first)
        
        logger.info(f'item vocab size : {len(item_vocab)}')
        logger.info(f'top 15 item vocab : {item_vocab[:15]}')
        #save item vocab
        save_vocab_file(item_vocab,data_path,args.str_vocab_file)
        
        logger.info('Start to create item_vocab embed')
        #create item embed vector
        create_embeds(item_token2id,kv_model,data_path,args)
    


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default=r'.\Data',help='Root dir for save data')
    parser.add_argument('--task',type=str,required=True,choices=['fakeJob'],help='The training Model task.')
    parser.add_argument('--mode',type=str,required=True,choices=['train','test'],
                        help='Determined use train set or test set.')

    parser.add_argument('--str_vocab_file',type=str,default='item_vocab.txt',help='The file name for save item vocab')
    parser.add_argument('--context_vocab_file',type=str,default='jobText_vocab.txt',help='The file name for save context vocab.')
    parser.add_argument('--lda_vocab_file',type=str,default=r'.\model\topic_model\lda_vocab.pkl',
                        help='The file name for lda vocab')
    parser.add_argument('--max_item_size',type=int,default=30000,help='The max vocab size for title.')
    parser.add_argument('--min_item_freq',type=int,default=5,help='The minimum frequency needed to include a token in title vocab.')
    parser.add_argument('--max_context_vocab',type=int,default=30000,help='Max vocab size for context word')
    parser.add_argument('--spec_first',action='store_true',
                        help='Whether to add special tokens into the vocabulary at first.If not call,special token will add into last')
    
    parser.add_argument('--select_context_name',type=str,nargs='+',choices=['company_profile','description','requirements','benefits'],
                       default=None,help='Select data column for building context vocab.')
    parser.add_argument('--select_item_name',type=str,default=None,help='select data coulmn for build item vocab')

    parser.add_argument('--encode_format',type=str,default='utf-8',help="The read data's encoding format")
    parser.add_argument('--nlp_type',type=str,default='en_core_web_md',help='The model name for nlp process.')
    parser.add_argument('--model_type',type=str,default='orig',help='The model name for pretrain embed model.')
    parser.add_argument('--corpora',type=str,default='wiki',help='Corpora name for train word embed.')

    parser.add_argument('--embed_dim',type=int,default=300,help='The pretrain embed model dim.')
    parser.add_argument('--embed_type',type=str,default='skg',choices=['skg','cbw'],help='The class of embed model.')

    args=parser.parse_args()
    set_log_config()
    main(args)
