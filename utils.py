import pickle
import os
import logging
import json

from #import model and build
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
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,args.item_vocab_file),'r',encoding='utf-8')]

def load_text_vocab(args):
    return [w.strip() for w in open(os.path.join(args.data_dir,args.task,args.text_vocab_file),'r',encoding='utf-8')]

def load_edu_dict(args):
    with open(os.path.join(args.saved_dir,args.process_dir,r'edu_level.json'),'r',encoding='utf-8') as f_r:
        edu_dict=json.load(f_r)
    return edu_dict

def load_job_dict(args):
    with open(os.path.join(args.saved_dir,args.process_dir,r'job_level.json'),'r',encoding='utf-8') as f_r:
        job_dict=json.load(f_r)
    return job_dict

