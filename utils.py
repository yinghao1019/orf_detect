import pickle
import os
import logging
import json
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,balanced_accuracy_score

def load_special_tokens(args):
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,'special_tokens.txt'))]

def set_log_config():
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

def load_ftQuery(args):
    with open(os.path.join(args.data_dir,args.task,r'fastText/wiki-news-querys.json'),'r',encoding='utf-8') as f_r:
        ft_query=json.load(f_r)
    return ft_query

def load_item_vocab(args):
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,'item_vocab.txt'),'r',encoding='utf-8')]

def load_text_vocab(args):
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,'jobText_vocab.txt'),'r',encoding='utf-8')]

def load_edu_dict(args):
    with open(os.path.join(args.data_dir,args.task,r'edu_level.json'),'r',encoding='utf-8') as f_r:
        edu_dict=json.load(f_r)
    return edu_dict

def load_job_dict(args):
    with open(os.path.join(args.data_dir,args.task,r'job_level.json'),'r',encoding='utf-8') as f_r:
        job_dict=json.load(f_r)
    return job_dict

def get_metrics(y_predict,y_true):
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