from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,balanced_accuracy_score
from model.modeling_orfRnn import RnnFakeDetector
from model.modeling_orfBert import Bert_Fakedetector
from model.utils_model import model_uniConfig
from transformers import BertConfig,BertTokenizer

import pickle
import os
import logging
import json
import random
import numpy as np
import torch


def load_special_tokens(args):
    """
    This func read speical_tokens.txt in Data/fakejob to load NER tag list.

    Args:
     args(ArgumentParser object):This args object should contain data_dir,task attr.
    
    Returns:
      list of str:the token for NER
    """
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,'special_tokens.txt'))]

def set_log_config():
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

def load_item_vocab(args):
    """
    This func read item_vocab.txt in Data/fakejob to load title's vocab of ORF datasets.

    Args:
     args(ArgumentParser object):This args object should contain data_dir,task attr.
    
    Returns:
      list of str: the token for title's vocab
    """
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,'item_vocab.txt'),'r',encoding='utf-8')]

def load_text_vocab(args):
    """
    This func read jobText_vocab.txt in Data/fakejob to load text vocab of ORF datasets.

    Args:
     args(ArgumentParser object):This args object should contain data_dir,task attr.
    
    Returns:
      list of str: the token for text's vocab that contain desc,requirement,profile,benefits
    """
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,'jobText_vocab.txt'),'r',encoding='utf-8')]

def load_edu_dict(args):
    """
    This func read edu_level.json in Data/fakejob to load edu mapping dict.

    Args:
     args(ArgumentParser object):This args object should contain data_dir,task attr.
    
    Returns:
      dict : The index mapping for edu level name. 
    """
    with open(os.path.join(args.data_dir,args.task,r'edu_level.json'),'r',encoding='utf-8') as f_r:
        edu_dict=json.load(f_r)
    return edu_dict

def load_job_dict(args):
    """
    This func read job_level.json in Data/fakejob to load edu mapping dict.

    Args:
     args(ArgumentParser object):This args object should contain data_dir,task attr.
    
    Returns:
      dict : The index mapping for job level name. 
    """
    with open(os.path.join(args.data_dir,args.task,r'job_level.json'),'r',encoding='utf-8') as f_r:
        job_dict=json.load(f_r)
    return job_dict

def get_metrics(y_predict,y_true):
    """
    Compute model metric of binary class output using skit-learn.
    The metric contain Precision,Recall,TNR,f_score,balance_acc

    Args:
     y_predict(numpy array):
     y_true(numpy array):

    Returns:
     dict of metrics: The metrics type contain precision,recall,f_score,balance_acc,TNR
    """
    assert len(y_predict)==len(y_true)

    #compute TNR & TPR 
    tn, fp, fn, tp=confusion_matrix(y_true,y_predict).ravel()
    TNR=tn/(tn+fn)
    precision=precision_score(y_true,y_predict)
    recall=recall_score(y_true,y_predict)
    f_metrics=f1_score(y_true,y_predict)
    balance_acc=balanced_accuracy_score(y_true,y_predict)
    metrics={'precision':precision,'TNR':TNR,'recall':recall,'f_score':f_metrics,'balance_acc':balance_acc}
    return metrics
def set_rndSeed(args):
    """
    Set rnadom seed to control random(builtin,numpy,torch,cuda,cudnn)

    Args:
     args(ArgumentParser object):This args object should contain random_seed     
    """
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.eterministic=True
def count_parameters(m):
    """
    Compute model total trainable params
    Args:
     m(torch.nn.Module object): the model class which inherient from torch.nn.module
    
    Returns:
     int:the total numbers of model's param 
    """
    return sum([p.numel() for p in m.parameters() if p.requires_grad])

#Use for loading trained model
MODEL_CLASSES={
    'bert-base-uncased':(model_uniConfig,BertTokenizer,Bert_Fakedetector),
    'gru-attn':(model_uniConfig,load_text_vocab,RnnFakeDetector),
}
MODEL_PATH={
    'bert-dnn':'bert-base-uncased',
    'rnn-dnn':'gru-attn'
}