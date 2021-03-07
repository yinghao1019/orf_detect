import pandas as pd
import numpy as np
import logging
import spacy
import torch
import copy
import json
import os

from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2dense
from process.text_process import sent_process,text_clean,count_links,remove_stopwords,doc_process
from utils import load_edu_dict,load_job_dict,load_special_tokens,load_text_vocab,load_item_vocab
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import namedtuple
import torch
#create logger
logger=logging.getLogger(__name__)
#define features field
InputExample=namedtuple('InputExample',['cp_file','desc','require','benefits','title','has_link',
                                        'has_logo','has_remote','job_level','edu_level','label'])
InputFeature=namedtuple('InputFeature',['job_context','job_segment','cp_context','cp_segment',
                                        'item_context','meta_data','label'])
class FakeJobDataset(Dataset):
    def __init__(self,examples):
        self.data=examples
    def __getitem__(self,key):
        #fetch data
        example=self.data[key]

        #seperate each data
        job_context=example.job_context
        job_segment=example.job_segment
        cp_context=example.cp_context
        cp_segment=example.cp_segment

        item_context=example.item_context
        meta_data=example.meta_data

        label=example.label

        return (job_context,job_segment,cp_context,cp_segment,item_context,meta_data,label)

    def __len__(self):
        return len(self.data)


class process_data:
    def __init__(self,data,args):
        self.data=data
        self.job_level=load_job_dict(args)
        self.edu_level=load_edu_dict(args)
        self.lda_vocab=Dictionary.load(os.path.join(args.saved_dir,args.process_dir,args.topic_model,args.lda_vocab))
        self.lda_model=LdaMulticore.load(args.ldaModel_file)
        #load nlp piepline model
        spacy.require_gpu()
        self.nlp_pipe=spacy.load(args.nlp_model)
        self.args=args
    def __iter__(self):
        for idx,data in enumerate(self.data.itertuples(name='example')):
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
        cp_profile=text_clean(data.company_profile) if not np.isnan(data.company_profile) else ''
        desc=text_clean(data.description) if not np.isnan(data.description) else ''
        requires=text_clean(data.requirements) if not np.isnan(data.requirements) else ''
        benefits=text_clean(data.benefits) if not np.isnan(data.benefits) else ''
        title=text_clean(data.title)

        #create feature
        has_descLink=1 if count_links(desc)!=0 else 0
        require_edu=self.edu_level[data.required_education] if data.required_education in self.edu_level else 0
        require_job=self.job_level[data.required_experience] if data.required_experience in self.job_level else 0

        #tokenized text
        cp_profile=[remove_stopwords(sent_process(sent)) for sent in self.nlp_pipe(cp_profile).sents] if cp_profile else []
        desc=[remove_stopwords(sent_process(sent)) for sent in self.nlp_pipe(desc).sents] if desc else []
        requires=[remove_stopwords(sent_process(sent)) for sent in self.nlp_pipe(requires).sents] if requires else []
        benefits=[remove_stopwords(sent_process(sent)) for sent in self.nlp_pipe(benefits).sents] if benefits else []
        title=doc_process(self.nlp_pipe(title))
        
        return InputExample(cp_file=cp_profile,desc=desc,require=requires,benefits=benefits,title=title,
                            has_link=has_descLink,has_logo=data.has_company_logo,has_remote=data.telecommuting,
                            job_level=require_job,edu_level=require_edu,label=data.fraudulent)



def convert_BertFeature(data,tokenizer,cp_sentNum,desc_sentNum,
                        require_sentNum,benefit_sentNum,max_textLen,
                        cls_seg=0):
    #build special tokens
    unk_token=tokenizer.unk_token
    sep_token=tokenizer.sep_token
    cls_token=tokenizer.cls_token
    column_sents={'job':[('desc',desc_sentNum),('require',require_sentNum)],
                 'cp':[('cp_file',cp_sentNum),('benefits',benefit_sentNum)],
                 }
    #build dict for saving each data
    save_dict={}

    #max_len minus 1 because consider text containing cls token
    max_textLen-=1

    for f_name in ['job','cp']:
        #init token ids
        tokens=[cls_token]
        segment_ids=[cls_seg]

        for seg_id,(c_name,max_sent) in enumerate(columns[f_name]):
            #build subword
            subwords=[]
            doc=getattr(data,c_name)#loading data

            if doc:
                for sent in doc[:max_sent]:
                    for w in sent:
                        subword=tokenizer.tokenize(w)

                        if not subword:
                            subword=[unk_token]
                        
                        subwords.extend(subword)
            
            #trunked subword len
            max_seqLen=max_textLen//2
            subwords=subwords[:max_seqLen-1]
            #add sep_token
            subwords.extend([sep_token])
            #build segment id
            tokens.extend(subwords)
            segment_ids.extend([seg_id]*len(subwords))

        #convert to token_ids
        token_ids=tokenizer.convert_tokens_to_ids(tokens)
        
        assert len(token_ids)==len(segment_ids)

        #save token_id,segment for field
        save_dict[f_name+'_context']=token_ids
        save_dict[f_name+'_segment']=segment_ids

    return InputFeature(**save_dict)

def convert_RnnFeature(data,vocab,cp_sentNum,desc_sentNum,
                    require_sentNum,benefit_sentNum,max_textLen):
    #build dict for saving each data                
    save_dict={}
    column_sents={'job':[('desc',desc_sentNum),('require',require_sentNum)],
                 'cp':[('cp_file',cp_sentNum),('benefits',benefit_sentNum)],
                 }
    #max_len minus 1 because consider text containing cls token
    max_textLen-=1
    max_docLen=max_textLen//2

    for f_name in list(column_sents.keys()):
        #init token ids
        text_ids=[]

        for idx,(c_name,max_sent) in enumerate(column_sents[f_name]):
            
            #build subword
            doc_tokens=[]
            doc=getattr(data,c_name)

            if doc:
                for sent in doc[:max_sent]:
                    sent_tokens=[]
                    for w in sent:
                        token_id=vocab.index(w) if w in vocab else vocab.index('UNK')
                        sent_tokens.append(token_id)
                    sent_tokens+=[vocab.index('[SEP]')]

                    doc_tokens.extend(sent_tokens)

                #truncated doc length
                doc_tokens=doc_tokens[:max_docLen]

                #replace last token to [SEP] token
                doc_tokens[-1]=vocab.index('[SEP]')  if doc_tokens[-1]!=vocab.index('[SEP]')
            
            else:
                #filled empty string value
                doc_tokens.extend([vocab.index('PAD')]*max_docLen)

            if idx==0:
                doc_tokens+=[vocab.index('[CONTEXT]')]

            text_ids.extend(doc_tokens)#combined text

        assert len(text_ids)<=max_textLen

        #saved contexts
        save_dict[f_name+'_context']=text_ids
    
    return InputFeature(**save_dict)

text_extractor={'bert':convert_BertFeature,'rnn':convert_RnnFeature}

class transform_feature:
    def __init__(self,text_type,lda_vocab_path,lda_model_path,args):
        self.convert_text=text_extractor[text_type]
        self.item_vocab=load_item_vocab(args)

        #read lda_vocab & model
        self.lda_vocab=Dictionary.load(lda_vocab_path)
        self.lda_model=LdaMulticore.load(lda_model_path)

    def convert_all_data(self,data,tokenizer,cp_maxNum,desc_maxNum,
                         require_maxNum,benefits_maxNum,max_textLen):
        all_data=[]
        for idx,example in enumerate(data):

            if idx%500==0:
                logger.info(f'Already convert {idx} nums feature!')
                logger.info('****display example****')
                logger.info(all_data[idx-1])
            #extract & convert job text
            feature=self.convert_text(example,tokenizer,cp_maxNum,desc_maxNum,
                                      require_maxNum,benefits_maxNum,max_textLen)

            #combine meta data
            meta_data=[]
            for f_name in ['has_link','has_logo','has_remote','job_level','edu_level']:
                meta_data.append(getattr(data,f_name))

            #extract meta data(word num,is empty,topics)
            for data in self.extract_metadata(example):
                meta_data.extend(data)
            
            #convert to ids
            item_tokenIds=[self.item_vocab.index(w) for w in example.title if w in self.item_vocab else 
                           self.item_vocab.index('UNK')]
            
            
            #set data attr
            feature.meta_data=meta_data
            feature.item_context=item_tokenIds
            feature.label=example.label

            all_data.append(feature)

        return all_data
            

    def extract_metadata(self,example):
        #extract topics
        desc_doc=[w for sent in example.desc for w in sent] if example.desc else []
        desc_bow=self.lda_vocab(desc_doc)
        desc_topics=self.lda_model.get_document_topics(desc_bow)
        desc_topics=corpus2dense([desc_topics],num_terms=self.lda_model.num_topics,num_docs=1).tolist()[0]

        if not desc_topics:
            desc_topics.extend([0.0]*self.lda_model.num_topics)

        #compute each text total words
        job_fields_wordN=[]
        for f_name in ['cp_file','require','benefits']:
            if getattr(example,f_name):
                words=[w for sent in getattr(example,f_name) for w in sent]
                wordN=len(words)
            else:
                wordN=0
            job_fields_wordN.append(wordN)
        
        job_fields_wordN.insert(1,len(desc_doc))

        #detect whether text is empty or not
        cp_empty=1 if not example.cp_file else 0
        require_empty=1 if not example.requires else 0


        return job_fields_wordN,[cp_empty,require_empty],desc_topics

def create_min_batch(tensors):
    #get & convert to tensor
    job_tensors=[d[0] for d in tensors]
    cp_tensors=[d[2] for d in tensors]
    item_tensors=[d[4] for d in tensors]
    metaData_tensors=torch.stack([d[5] for d in tensors])
    label_tensors=torch.stack([d[6] for d in tensors])

    #pad  & transform tensors
    job_tensors=pad_sequence(job_tokens,batch_first=True)
    cp_tensors=pad_sequence(cp_tokens,batch_first=True)


    if tensors[0][1]:

        #pad  & transform tensors
        jobSegment_tensors=[d[1] for d in tensors]
        cpSegment_tensors=[d[3] for d in tensors]
        jobSegment_tensors=pad_sequence(jobSegment_tensors,batch_first=True)
        cpSegment_tensors=pad_sequence(cpSegment_tensors,batch_first=True)

        
        #create mask tensors
        jobMask_tensors=torch.ones(job_tensors.size(),dtype=torch.long).masked_fill_(job_tensors==0,0)
        cpMask_tensors=torch.ones(job_tensors.size(),dtype=torch.long).masked_fill_(job_tensors==0,0)

        return {'job_token_tensors':job_tensors,'job_seg_tensors':jobSegment_tensors,'job_mask_tensors':jobMask_tensors,
                'cp_token_tensors':cp_tensors,'cp_seg_tensors':cpSegment_tensors,'cp_mask_tensors':cpMask_tensors,
                'item_tensors':item_tensors,'metaData_tensors':metaData_tensors,'label_tensors':label_tensors,}
    else:
        return {'job_token_tensors':job_tensors,'cp_token_tensors':cp_tensors,'item_tensors':item_tensors,
                'metaData_tensors':metaData_tensors,'label_tensors':label_tensors,}


def load_and _cacheEamxples(args,tokenizer,mode):
    #build file path
    file_path=os.path.join(args.data_dir,args.task,
                           'cached_{}_{}_{}_{}.zip'.format(
                               args.task,args.mode,
                               args.max_textLen,
                               list(filter(None,args.model_name_or_path.split('/'))).pop(-1)))
    
    if os.path.isfile(file_path):
        logger.info(f'Loading feature from {file_path}')
        datasets=torch.load(file_path)
    else:
        logger.info(f'Build {mode} dataset!')
        #read
        datasets=pd.read_csv(os.path.join(args.data_dir,args.task,mode,'data.csv'),encoding='utf-8')

        #text processing
        logger.info('Start data process!')
        datasets=process_data(datasets,args)

        #extract feature
        if args.model_type.endswith('bert'):
            data_transformer=transform_feature('bert',args.lda_vocab_path,args.lda_model_path,args)
        elif args.model_type.endswith('rnn'):
            data_transformer=transform_feature('rnn',args.lda_vocab_path,args.lda_model_path,args)
        
        logger.info('Start to transform to datasets!')
        datasets=data_transformer.convert_text(datasets,tokenizer,args.cp_sentNum,args.desc_sentNum,
                                                args.require_sentNum,args.benefit_sentNum,args.max_textLen)

        #save data to disk
        torch.save(datasets,file_path)
        logger.info(f'Save data to {file_path} success!')

    #convert to tensor datasets
    datasets=FakeJobDataset(datasets)
    logger.info('Convert to tensor dataset success!')

    return datasets


        




