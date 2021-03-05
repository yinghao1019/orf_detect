import pandas as pd
import numpy as np
import gensim 
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models import ldamulticore,coherencemodel
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import logging
import os
import argparse
from text_process import corpus_process
import sys
sys.path.append(os.getcwd())
#customize module
from utils import load_special_tokens,set_log_config,load_ftQuery

logger = logging.getLogger(__name__)
def choose_topic(bow_corpus,corpus,topics,vocab,chunksize, passes,eval_mode,workers):
    #build Model &train
    lda_model=ldamulticore.LdaMulticore(bow_corpus,topics,id2word=vocab,workers=workers,
                                chunksize=chunksize,passes=passes)
    #evaluate
    scores=[]
    for evals in eval_mode:
        cm=coherencemodel.CoherenceModel(lda_model,texts=corpus,coherence=evals,dictionary=vocab)
        scores.append(cm.get_coherence())#get score

    return lda_model,scores
    
def plot_scores(topics,coherence_scores,colors,markers,eval_modes):
    #build figure
    fig=plt.figure(figsize=(10,8))
    n_rows=len(coherence_scores[0])
    coherence_scores=zip(*coherence_scores)
    #display score
    for  idx,(s,mode) in enumerate(zip(coherence_scores,eval_modes)):
        ax=fig.add_subplot(n_rows,1,idx+1)
        ax.plot(list(topics),list(s),color=colors,lw=1,ms=2)
        ax.set_xlabel('Topics Num')
        ax.set_ylabel(f'Topic coherence score : {mode}')
        ax.set_title(f'The {mode} score for evaluating topic Model')
    fig.tight_layout()
    plt.show()
    # fig.tight_layout()
def main(args):
    data_path=os.path.join(args.data_dir,args.task)
    save_path=os.path.join(args.save_dir,args.model_class)
    #loading vocab dictionary
    context_vocab=Dictionary.load(os.path.join(save_path,args.vocab_file))
    #loading model
    spacy.require_gpu()
    en_nlp=spacy.load('en_core_web_md')
    #read_data
    corpus=pd.read_csv(os.path.join(data_path,args.mode,'data.csv'),encoding='utf-8')
    logger.info('Read Data success,Start to process data...')
    #selected columns
    corpus=corpus.loc[:,args.select_column].dropna().values.tolist()
    #corpus process
    corpus=corpus_process(corpus,en_nlp)
    logger.info('Process data success!')
    #convert to bow
    corpus_bow=[context_vocab.doc2bow(doc) for doc in corpus]
    #save corpus of bow format
    gensim.corpora.mmcorpus.MmCorpus.serialize(os.path.join(save_path,args.bow_file),corpus_bow,context_vocab)
    logger.info(f'save corpus of bow format to {save_path} success!')

    if args.do_select:
        logger.info('Start to select best topic num!')
        topic_ranges=range(args.min_topic,args.max_topic)
        topics_score=[]

        for  topic in topic_ranges:
            _,scores=choose_topic(corpus_bow,corpus,topic,context_vocab,args.bs,args.epoch,args.eval_modes,args.num_workers)
            topics_score.append(scores)

            #display model info
            logger.info(f'evaluate topic model quality for {topic} topics')
            for n,s in zip(args.eval_modes,scores):
                logger.info(f"topic model's {n}:{s}")

        #build dataframe to display
        score_df=pd.DataFrame(topics_score,index=topic_ranges,columns=args.eval_modes)
        logger.info(f'topic model total metrics:{score_df}')
        #using line plot to display
        plot_scores(topic_ranges,topics_score,'r','o',args.eval_modes)
    
    if args.do_train:
        logger.info(f'LDA model training for {args.best_topic}!')
        lda_model,_=choose_topic(corpus_bow,corpus,args.best_topic,context_vocab,args.bs,args.epoch,args.eval_modes,args.num_workers)

        logger.info(f'Show lda model topic distribution')
        logger.info(f'{lda_model.show_topics()}')
        #save model
        try:
            lda_model.save(os.path.join(save_path,args.lda_file_path))
            logger.info(f'Save lda model success to {args.lda_file_path}')
        
        except:
            logger.info('lda_model saved error!')

if __name__=='__main__':
    parser=argparse.ArgumentParser(usage='%(prog)s [-h] [--eval_mode [{u_mass,c_v,c_uci,c_npmi} ...]] [--do_select] \n \
                                  [--do_train] [--min_topic MIN_TOPIC] \n \
                                  [--max_topic MAX_TOPIC] [--best_topic BEST_TOPIC] \n \
                                  [--select_column [{company_profile,description,requirements,benefits}...]]')

    parser.add_argument('--data_dir',type=str,default=r'.\Data',help='Root dir for save data.')
    parser.add_argument('--task',type=str,default='fakeJob',help='The training Model task.')
    parser.add_argument('--mode',type=str,default='train',help='The data class.')
    parser.add_argument('--lda_file_path',type=str,default='lda_model',help='The prefix path name for saveing topic model.')
    parser.add_argument('--save_dir',type=str,default=r'.\saved_model\process_model',help='The parent dir for save topic model.')
    parser.add_argument('--model_class',type=str,default='topic_model',help='')
    parser.add_argument('--vocab_file',type=str,default='lda_vocab.pkl',help='The vocabulary file path for saved model')
    parser.add_argument('--bow_file',type=str,default='lda_corpus.mm',help='The file path for bow corpus')
    parser.add_argument('--select_column',type=str,nargs='+',required=True,
                        choices=['company_profile','description','requirements','benefits'],
                        help='Select data column for building topic model.')
    
    
    parser.add_argument('--num_workers',type=int,default=os.cpu_count(),
                        help='Determined cpu core nums to training model.Default is os.cpu_count().')
    parser.add_argument('--epoch',type=int,default=10,help="Train times for training model.Default is 10.")
    parser.add_argument('--bs',type=int,default=128,help="Batch size for training model.Default is 128.")
    parser.add_argument('--min_topic',type=int,default=2,
                        help="The minimum topic for select best topic Model.Default is 3.")
    parser.add_argument('--max_topic',type=int,default=15,
                        help="The maximum topic for select best topic Model.Default is 15")
    parser.add_argument('--best_topic',type=int,default=4,help="Determined topic nums for train model.Default is 4.")

    parser.add_argument('--eval_modes',type=str,nargs='+',default=['u_mass'],
                        choices=['u_mass', 'c_v', 'c_uci', 'c_npmi'],help='Evaluate method for different topic nums model.')

    parser.add_argument('--do_select',action='store_true',help='Determined evaluate model or not.')
    parser.add_argument('--do_train',action='store_true',help='Determined train model or not.')
    args=parser.parse_args()
    #set program logging level
    set_log_config()
    main(args)
    
