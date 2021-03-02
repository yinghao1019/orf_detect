import pandas as pd
import numpy as np
import spacy
import os
import logging
import linecache
import argparse
from collections import Counter
from gensim.corpora.dictionary import Dictionary
import sys
sys.path.append(os.getcwd())# add path to module search path
#customize module
from utils import load_special_tokens,set_log_config,load_ftQuery
from text_process import corpus_process,en_stopwords

logger = logging.getLogger(__name__)

'''Vocab process funcs'''
def combine_data(data,col_names):
    # combined data non na data
    nonZero_df=data.loc[:,col_names].copy().notna().values.reshape(-1)
    df=data.loc[:,col_names].values.reshape(-1)
    return df[nonZero_df]



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
    num,dim=map(int,linecache.getline(embed_path,1).strip().split())
    #get vocab nums
    vocab_num=len(vocab.keys())
    #build storage vocab embed array
    vocab_embed=np.zeros((vocab_num,dim))

    logger.info(f'pretrain word embed nums:{num} dim:{dim}')
    logger.info(f'vocab nums:{vocab_num} \t top5 word pair:{list(vocab.items())[:5]}')
    logger.info(f'Embed vector shape:{vocab_embed.shape}')
    #get pretrain emebd_dim
    try:
        undefine_w=[]
        item=0
        for w,w_id in vocab.items():
            if w in query:
                #get word data in pretrain file
                vector=linecache.getline(embed_path,int(query[w])+1).strip().split(' ')[1:]
                vocab_embed[w_id]=np.array(list(map(float,vector)))
            else:
                vocab_embed[w_id]=np.random.randn(dim)
                undefine_w.append(w)
            
            item+=1
            if item%500==0:
                logger.info(f'Already convert {item} word to embed success!')
        logger.info('show pretrain embed:\n{}'.format(vocab_embed[20:23]))
        logger.info('Undefined word num:{}'.format(len(undefine_w)))    
    
    except:
        logger.info('Loading pretrain embed error!')
        print('Embed_path:',embed_path)
            

    #save vector to disk
    save_dir=os.path.join(args.data_dir,args.task)
    #detect vocab dir
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(save_dir)
        logger.info(f'{save_dir} not exists!Create new one')

    try:
        #write undefine w to disk
        with open(os.path.join(save_dir,f"{vocab_num}_undefined_word.txt"),'w',encoding='utf-8') as f_w:
            for u_w in undefine_w:
                f_w.write(u_w+'\n')
        logger.info('Save undefined word success!')

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
    #load nlp model
    spacy.require_gpu()
    en_nlp=spacy.load(args.model_type)
    
    # read data
    df = pd.read_csv(os.path.join(data_path,args.mode,'data.csv'),encoding=args.encode_format)
    special_tokens=load_special_tokens(args)
    pretrain_query=load_ftQuery(args)#load fastText index query

    df = df.drop_duplicates(subset=['description','title'],keep=False)
    logger.info(f'Read data from {os.path.join(data_path,args.mode)} success!')
    logger.info(f'df shape:{df.shape}')
    logger.info(f'Special token: {special_tokens} nums: {len(special_tokens)}')

    #selected columns and build corpus with using nlp pipe
    #corpus shape = list of list of token
    logger.info('****Start to build vocab****')

    if args.select_context_name:

        logger.info(f'Start to build context vocab for {args.select_context_name}!')

        #data prepare
        context_data=combine_data(df,args.select_context_name).tolist()
        context_corpus=corpus_process(context_data,en_nlp) 
        # build lda model vocab
        context_dict=Dictionary(context_corpus)
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

        logger.info('Create pretrain embed vector..')

        create_embeds(context_dict.token2id,pretrain_query,embed_path,args)

    if args.select_item_name:
        
        logger.info(f'Start to build string vocab for {args.select_item_name}!')

        #data prepare
        item_data=combine_data(df,args.select_item_name).tolist()
        item_corpus=corpus_process(item_data,en_nlp)

        #build title columns vocab
        item_corpus=[w for doc in item_corpus for w in doc]
        item_vocab,item_token2id=build_vocab(item_corpus,special_tokens,args.max_item_size,
                                               args.min_item_freq,args.spec_first)
        
        logger.info(f'item vocab size : {len(item_vocab)}')
        logger.info(f'top 15 item vocab : {item_vocab[:15]}')
        #save item vocab
        save_vocab_file(item_vocab,data_path,args.str_vocab_file)

        logger.info('Create pretrain embed vector..')
        
        #create item embed vector
        create_embeds(item_token2id,pretrain_query,embed_path,args)
    


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default=r'.\Data',help='Root dir for save data')
    parser.add_argument('--task',type=str,default=None,required=True,choices=['fakeJob'],help='The training Model task.')
    parser.add_argument('--mode',type=str,default=None,required=True,choices=['train','test'],
                        help='Determined use train set or test set.')

    parser.add_argument('--str_vocab_file',type=str,default='str_vocab.txt',help='The file name for save string vocab')
    parser.add_argument('--context_vocab_file',type=str,default='context_vocab.txt',help='The file name for save context vocab.')
    parser.add_argument('--lda_vocab_file',type=str,default=r'.\saved_model\process_model\topic_model\lda_vocab.pkl',
                        help='The file name for lda vocab')
    parser.add_argument('--pretrain_embed_file',type=str,default=r'fastText\wiki-news-300d-1M-subword.vec',
                        help='The path for fastText embedding file.')

    parser.add_argument('--max_item_size',type=int,default=30000,help='The max vocab size for title.')
    parser.add_argument('--min_item_freq',type=int,default=3,help='The minimum frequency needed to include a token in title vocab.')
    parser.add_argument('--max_context_vocab',type=int,default=30000,help='Max vocab size for context word')
    parser.add_argument('--spec_first',action='store_true',
                        help='Whether to add special tokens into the vocabulary at first.If not call,special token will add into last')
    
    parser.add_argument('--select_context_name',type=str,nargs='+',choices=['company_profile','description','requirements','benefits'],
                       default=None,help='Select data column for building context vocab.')
    parser.add_argument('--select_item_name',type=str,default=None,help='select data coulmn for build item vocab')

    parser.add_argument('--encode_format',type=str,default='utf-8',help="The read data's encoding format")
    parser.add_argument('--model_type',type=str,default='en_core_web_md',help='The model name for nlp process.')

    args=parser.parse_args()
    set_log_config()
    main(args)
