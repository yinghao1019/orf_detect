from data_loader import load_and_cacheEamxples,create_mini_batch
from model_trainer import Train_pipe
from utils import *
from torch.utils.data import DataLoader
import argparse
import logging
import os
logger=logging.getLogger(__name__)

def main(args):
    #load vocab
    items_vocab=load_item_vocab(args)
    spec_vocab=load_special_tokens(args)

    logger.info('Load model data...')
    #load model & tokenizer
    if args.used_model.startswith('bert'):
        pretrained_name=MODEL_PATH[args.used_model]#get pretrain type
        tokenizer=MODEL_CLASSES[pretrained_name][1].from_pretrained(pretrained_name)
        

        uni_config=MODEL_CLASSES[pretrained_name][0]
        uni_config['item_input_dim']=len(items_vocab)
        uni_config['pos_weight']=args.pos_weights
        uni_config['pretrain_path']=pretrained_name
        
        #add spec vocab to extend vocab num
        logger.info(f'Vocab num that before add new spec token:{len(tokenizer)}')
        tokenizer.additional_special_tokens=spec_vocab
        tokenizer.add_tokens(spec_vocab,special_tokens=True)
        # uni_config['vocab_num']=len(tokenizer)
        logger.info(f'Added vocab:{tokenizer.get_added_vocab()}')
        logger.info(f'Vocab num that after add new spec token:{len(tokenizer)}')
        #extend model input dim
        uni_config['vocab_num']=len(tokenizer)
        #build model
        model=MODEL_CLASSES[pretrained_name][2](**uni_config)

    elif args.used_model.startswith('rnn'):

        pretrained_name=MODEL_PATH[args.used_model]#get pretrain type
        uni_config=MODEL_CLASSES[pretrained_name][0]
        tokenizer=MODEL_CLASSES[pretrained_name][1](args)
        #insert vocab nums
        uni_config['text_input_dim']=len(tokenizer)
        uni_config['item_input_dim']=len(items_vocab)
        uni_config['pos_weight']=args.pos_weights
        #build model
        model=MODEL_CLASSES[pretrained_name][2](**uni_config)
        
    logger.info('Start to loading data !')
    #loading data
    train_data=load_and_cacheEamxples(args,tokenizer,'train')
    val_data=load_and_cacheEamxples(args,tokenizer,'test')
    
    #training model
    if args.do_train:
        logger.info('Start to train model !')
        #build training pipeline
        pipe=Train_pipe(train_data,val_data,model,args)
        pipe.train_model(uni_config)

        #evalutae model performance separately
        pipe.eval_model(train_data)
    if args.do_eval:
        logger.info('Start to eval model !')
        pretrained_name=MODEL_PATH[args.used_model]
        pretrain_class=MODEL_CLASSES[pretrained_name][2]
        pipe=Train_pipe.load_model(pretrain_class,train_data,val_data,args)   
        pipe.eval_model(val_data)
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--data_dir',type=str,default='.\Data',
                        help='Root dir path for save data.Default .\Data')

    parser.add_argument('--saved_dir',type=str,default='saved_model',help='')
    parser.add_argument('--process_dir',type=str,default=r'.\process\model',help='')
    parser.add_argument('--lda_model_file',type=str,default='topic_model\lda_model',
                        help='File path for pretrain lda model.')
    parser.add_argument('--lda_vocab_file',type=str,default='topic_model\lda_vocab.pkl',
                        help='File path for pretrain lda vocab.')
    parser.add_argument('--train_model_dir',type=str,default=None,required=True,
                        help='Path to saved training model dir.')
    parser.add_argument('--task',type=str,default='fakeJob',
                        help='Select train Model task.Default is fakeJob.')
    parser.add_argument('--nlp_model',type=str,default='en_core_web_md',
                        help='The name for select spacy model.')

    parser.add_argument('--edu_threshold',type=int,default=3,help='Lower edu threshold.')
    parser.add_argument('--job_threshold',type=int,default=3,help='Lower job threshold.')
    parser.add_argument('--pos_weights',type=int,nargs='+',default=[4.0],help='a weight of positive examples.')
    parser.add_argument('--max_textLen',type=int,default=300,
                        help='Set max word num After tokenize text.')
    parser.add_argument('--cp_sentNum',type=int,default=7,
                        help='Set max seq len After tokenize company_profile.')
    parser.add_argument('--desc_sentNum',type=int,default=11,
                        help='Set max seq len After tokenize description.')
    parser.add_argument('--require_sentNum',type=int,default=5,
                        help='Set max seq len After tokenize require.')
    parser.add_argument('--benefit_sentNum',type=int,default=3,
                        help='Set max seq len After tokenize benefit.')

    
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate for Adam.')
    parser.add_argument('--weight_decay',type=float,default=0.0,help='weight decay for Adam.')
    parser.add_argument('--max_norm',type=float,default=1.0,
                        help='Max norm to avoid gradient exploding.Default is 1.')
    parser.add_argument('--prob',type=float,default=0.5,
                       help='The probability threshold for predict pos class.')

    parser.add_argument('--train_bs',type=int,default=32,help='Train model Batch size. Default is 32.')
    parser.add_argument('--val_bs',type=int,default=32,help='Eval model Batch size. Default is 32.')
    parser.add_argument('--epochs',type=int,default=10,
                        help='If>0:set number of train model epochs. Default is 10.')
    parser.add_argument('--total_steps',type=int,default=0,
                        help='If>0:set number of train model epochs. Default is 0.')
    parser.add_argument('--grad_accumulate_steps',type=int,default=1,
                        help='Number of update gradient to accumulate before update model.Default is 1.')
    parser.add_argument('--warm_steps',type=int,default=100,help='Linear Warm up steps.Default is 100.')
    parser.add_argument('--logging_steps',type=int,default=300,
                        help='Every X train step to logging model info. Default is 300.')
    parser.add_argument('--save_steps',type=int,default=300,
                        help='Every X train step to save model info. Default is 300.')
    
    parser.add_argument('--used_model',type=str,required=True,choices=['bert-dnn','rnn-dnn'],
                        help='Select model for training.')
    parser.add_argument('--do_train',action='store_true',help='Whether to train model or not.')
    parser.add_argument('--do_eval',action='store_true',help='Whether to eval model or not.')
    parser.add_argument('--random_seed',default=1234,help='set random seed.')
    args=parser.parse_args()
    args.model_name_or_path=MODEL_PATH[args.used_model]

    #set logger config
    set_log_config()
    set_rndSeed(args)
    main(args)






