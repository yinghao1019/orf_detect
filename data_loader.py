import pandas as pd
import numpy as np
import logging
import spacy
import string
import torch
import copy
import json
import os

from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2dense
from transformers import BertTokenizer
from process.text_process import sent_process,text_clean,count_links,remove_stopwords,doc_process
from utils import load_edu_dict,load_job_dict,load_special_tokens,load_text_vocab,load_item_vocab,set_log_config
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import namedtuple,defaultdict
import torch

#create logger
logger=logging.getLogger(__name__)

#define features field
InputFeature=namedtuple('InputFeature',['cp_file','desc','require','benefits',
                                        'title','meta_data','label'])
BertFeature=namedtuple('BertFeature',['job_tokens','cp_tokens','job_segs','cp_segs',
                                       'title','meta_data','label'])

class BertDataset(Dataset):
    def __init__(self,examples,tokenizer,lda_vocab_path,lda_model_path,args):
        self.data=examples
        self.tokenizer=tokenizer
        #add new special token
        self.spec_tokens=load_special_tokens(args)
        self.tokenizer.additional_special_tokens=self.spec_tokens
        self.tokenizer.add_tokens(self.spec_tokens)
        self.args=args
        self.item_vocab=load_item_vocab(args)
        self.lda_vocab=Dictionary.load(lda_vocab_path)
        self.lda_model=LdaMulticore.load(lda_model_path)

        self.sent_lim=[self.args.cp_sentNum,self.args.desc_sentNum,
                        self.args.require_sentNum,self.args.benefit_sentNum]
        self.text_fields=self.data[0]._fields[:4]
    def __getitem__(self,key):
        #fetch data
        example=self.data[key]
        
        #extract each data
        wordN=[]
        topics=[]
        allToken_ids=[]
        allSegment_ids=[]

        for idx,(doc,max_sent) in enumerate(zip(example[:4],self.sent_lim)):

            #count total word num for [cp_file,require]
            if idx%2!=0:
                word_num=len([w for sent in doc for w in sent])
                wordN.append(word_num)
            
            #tokenize subword
            subwords=[]
            if doc:
                for sent in doc[:max_sent]:
                    for w in sent:
                        subword=self.tokenizer.tokenize(w)
                        if not subword:
                            subword=[self.tokenizer.unk_token]
                        subwords.extend(subword)
            #truncated subwords equal to max textLen
            max_textLen=self.args.max_textLen-1
            if len(subwords)>max_textLen:
                subwords=subwords[:max_textLen]
            
            #add sep token & convert
            subwords+=[self.tokenizer.sep_token]
            token_ids=self.tokenizer.convert_tokens_to_ids(subwords)
            segment_ids=[0 if idx<2 else 1]*len(token_ids)

            allToken_ids.append(token_ids)
            allSegment_ids.append(segment_ids)
        
        #combined data
        job_tokens=[self.tokenizer.cls_token_id]+allToken_ids[1]+allToken_ids[2]
        cp_tokens=[self.tokenizer.cls_token_id]+allToken_ids[0]+allToken_ids[3]
        job_segs=[0]+allSegment_ids[1]+allSegment_ids[2]
        cp_segs=[0]+allSegment_ids[0]+allSegment_ids[3]

        # extract topics
        desc_bow=[w for sent in example[1] for w in sent]
        if desc_bow:
            desc_bow=self.lda_vocab.doc2bow(desc_bow)
            desc_topics=self.lda_model.get_document_topics(desc_bow)
            desc_topics=corpus2dense([desc_topics],num_terms=self.lda_model.num_topics,num_docs=1).T.tolist()[0]
        else:
            desc_topics=[0.0]*self.lda_model.num_topics

        #convert title to index
        if example.title:
            item=[self.item_vocab.index(w) if w in self.item_vocab else self.item_vocab.index('[UNK]') for w in example.title]
        else:
            item=[self.item_vocab.index('[PAD]')]

        #combine other meta data=[orig,wordN,topic_distri]
        meta_data=example.meta_data+wordN+desc_topics
        return BertFeature(job_tokens=torch.tensor(job_tokens),cp_tokens=torch.tensor(cp_tokens),
                           job_segs=torch.tensor(job_segs),cp_segs=torch.tensor(cp_segs),
                           title=torch.tensor(item),meta_data=torch.tensor(meta_data),
                           label=torch.tensor([example.label]))

    def __len__(self):
        return len(self.data)

class RnnDataset(Dataset):
    def __init__(self,examples,vocab,lda_vocab_path,lda_model_path,args):
        self.data=examples
        self.vocab=vocab
        self.args=args
        self.item_vocab=load_item_vocab(args)
        self.lda_vocab=Dictionary.load(lda_vocab_path)
        self.lda_model=LdaMulticore.load(lda_model_path)
        self.sent_lim=[self.args.cp_sentNum,self.args.desc_sentNum,
                        self.args.require_sentNum,self.args.benefit_sentNum]
    def __getitem__(self,key):
        #fetch data
        example=self.data[key]

        #extract each data
        wordN=[]
        allToken_ids=[]

        for idx,(doc,max_sent) in enumerate(zip(example[:4],self.sent_lim)):

            #count total word num for [cp_file,require]
            if idx%2!=0:
                word_num=len([w for sent in doc for w in sent])
                wordN.append(word_num)
            
            #tokenize subword
            token_ids=[]
            if doc:
                for sent in doc[:max_sent]:
                    for w in sent:
                        if w in self.vocab:
                            word_index=self.vocab.index(w)
                        else:
                            word_index=self.vocab.index('[UNK]')
                        token_ids.append(word_index)

                    #add sent bound
                    token_ids.append(self.vocab.index('[SEP]'))

            if len(token_ids)>self.args.max_textLen:
                token_ids=token_ids[:self.args.max_textLen]

            allToken_ids.append(torch.tensor(token_ids))

        #extract topics
        desc_bow=[w for sent in example[1] for w in sent]
        if desc_bow:
            desc_bow=self.lda_vocab.doc2bow(desc_bow)
            desc_topics=self.lda_model.get_document_topics(desc_bow)
            desc_topics=corpus2dense([desc_topics],num_terms=self.lda_model.num_topics,num_docs=1).T.tolist()[0]
        else:
            desc_topics=[0.0]*self.lda_model.num_topicsimport 

        #convert title to index
        if example.title:
            item=[self.item_vocab.index(w) if w in self.item_vocab else self.item_vocab.index('[UNK]') for w in example.title]
        else:
            item=[self.item_vocab.index('[PAD]')]
        #fetch other data
        meta_data=example.meta_data+wordN+desc_topics


        #combine data
        features=allToken_ids+[torch.tensor(item)]+[torch.tensor(meta_data)]+[torch.tensor([example.label])]

        #data memeber=(token_ids for 4 text fields,segment_ids for 4 text fields,
        #               title,metadata,label)
        return InputFeature._make(features)

    def __len__(self):
        return len(self.data)

class process_data:
    def __init__(self,data,args):
        self.data=data
        self.job_level=load_job_dict(args)
        self.edu_level=load_edu_dict(args)
        #load nlp piepline model
        spacy.require_gpu()
        self.nlp_pipe=spacy.load(args.nlp_model)
        self.args=args
    def __iter__(self):
        for idx,data in enumerate(self.data):
            if idx%1000==0:
                logger.info(f'Already clean & convert {idx} examples!')
            example=self.extract_data(data)
            
            #display extract data info
            if idx<5:
                logger.info(f"****{idx+1} th's example information")
                logger.info(example)

            yield example

    def extract_data(self,data):
        #clean text 
        cp_profile=text_clean(data.company_profile) if not pd.isna(data.company_profile) else ''
        desc=text_clean(data.description) if not pd.isna(data.description) else ''
        requires=text_clean(data.requirements) if not pd.isna(data.requirements) else ''
        benefits=text_clean(data.benefits) if not pd.isna(data.benefits) else ''
        title=text_clean(data.title)

        #create meta feature
        has_descLink=count_links(data.description)!=0 if not pd.isna(data.description) else 0
        require_edu=self.edu_level[data.required_education] if data.required_education in self.edu_level else 0
        require_job=self.job_level[data.required_experience] if data.required_experience in self.job_level else 0
        lower_edu=1 if 0<require_edu<self.args.edu_threshold else 0
        lower_job=1 if 0<require_job<self.args.job_threshold else 0
        meta_data=[has_descLink,require_edu,require_job,lower_edu,lower_job]
        meta_data+=[data.has_company_logo,data.telecommuting]
        #tokenized text
        cp_profile=doc_process(self.nlp_pipe(cp_profile.lower())) if cp_profile else []
        desc=doc_process(self.nlp_pipe(desc.lower())) if desc else []
        requires=doc_process(self.nlp_pipe(requires.lower())) if requires else []
        benefits=doc_process(self.nlp_pipe(benefits.lower())) if benefits else []
        title=[w.lower() for w in title.split() if (w not in string.punctuation) and w.isalpha()]
        
        return InputFeature(cp_file=cp_profile,desc=desc,require=requires,benefits=benefits,title=title,
                            meta_data=meta_data,label=int(data.fraudulent))

def create_mini_batch(batchs):
    #group each field for list of tensors
    field_names=batchs[0]._fields
    batchs=list(zip(*batchs))


    #pad tokens with first 4 field convert to tensor
    for idx,b in enumerate(batchs[:4]):
        batchs[idx]=pad_sequence(b,batch_first=True)
    #stack tensors
    for idx,t in enumerate(batchs[-2:]):
        batchs[-2+idx]=torch.stack(t)

    #create field dict
    batch_dict=dict(zip(field_names,batchs))
    #create tensors
    if 'job_segs' in field_names:
        for f,v in list(batch_dict.items())[:2]:
            masks=torch.ones_like(v).masked_fill_(v==0,0)#create mask tensors
            n=f.split('_')[0]
            batch_dict[n+'_masks']=masks
    
    return batch_dict
#truncate dataset
def balanced_process(dataset,class_weight,labels_name):
    #get each class data
    pos_dataset=dataset.query(f'{labels_name}==1')
    neg_dataset=dataset.query(f'{labels_name}==0')

    #get data num for each class
    pos_num=len(pos_dataset)
    neg_num=len(neg_dataset)
    logger.info('Positive example nums:{}'.format(len(pos_dataset)))
    logger.info('Negative example nums:{}'.format(len(neg_dataset)))
    #random sampling data from neg datasets According to pos_num*class_weight
    neg_dataset=neg_dataset.sample(n=pos_num*class_weight,axis=0)
    logger.info('Negative example nums after balanced:{}'.format(len(neg_dataset)))

    #combine  and shuffle data order
    dataset=neg_dataset.append(pos_dataset,ignore_index=True).sample(frac=1).reset_index(drop=True)

    return dataset
def load_and_cacheEamxples(args,tokenizer,mode):
    #build path variable
    data_path=os.path.join(args.data_dir,args.task)
    process_path=os.path.join(args.saved_dir,args.process_dir)
    lda_model_path=os.path.join(process_path,args.lda_model_file)
    lda_vocab_path=os.path.join(process_path,args.lda_vocab_file)

    #build file path
    file_path=os.path.join(args.data_dir,args.task,
                           'cached_{}_{}_process_data.zip'.format(
                            args.task,mode))
    
    if os.path.isfile(file_path):
        logger.info(f'Loading feature from {file_path}')
        datasets=torch.load(file_path)
    else:
        logger.info(f'Build {mode} dataset!')
        #read
        datasets=pd.read_csv(os.path.join(args.data_dir,args.task,mode,'data.csv'),encoding='utf-8')
        #external process for train dataset
        if mode=='train':
            #remove duplicate example
            logger.info(f'data nums Before remove duplicated:{len(datasets)}')
            datasets=datasets.drop_duplicates(subset=['title','description'],ignore_index=True)
            logger.info(f'data nums After remove duplicated:{len(datasets)}')

            #balanced each class propotions for dataset
            datasets=balanced_process(datasets,args.pos_weights[0],'fraudulent')

        datasets=datasets.itertuples()
        #text processing
        logger.info('Start data process!')
        datasets=list(process_data(datasets,args))

        # save data to disk
        torch.save(datasets,file_path)
        logger.info(f'Save data to {file_path} success!')
    
    #extract feature
    logger.info('Convert example to tensor data!')

    if args.used_model.startswith('bert'):
        datasets=BertDataset(datasets,tokenizer,lda_vocab_path,
                            lda_model_path,args)

    elif args.used_model.startswith('rnn'):
        logger.info('Using rnn datasets')
        datasets=RnnDataset(datasets,tokenizer,lda_vocab_path,
                            lda_model_path,args)

    logger.info(f'****Display example state of {mode} dataset****')
    for i in range(5):
        logger.info(f'{i} example: {datasets[int(i)]}')

    return datasets




