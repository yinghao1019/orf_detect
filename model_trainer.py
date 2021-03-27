from torch.utils.data import DataLoader,SequentialSampler,RandomSampler
from torch.optim import Adam
from torch import autograd
from utils import get_metrics
from data_loader import create_mini_batch
from transformers import get_linear_schedule_with_warmup,PreTrainedModel,AdamW

import numpy as np
import pandas as pd
import torch
import argparse
import os
import logging
from tqdm import tqdm,trange
logger=logging.getLogger(__name__)

class Train_pipe:
    def __init__(self,train_dataset,val_dataset,model,args):
        self.train_data=train_dataset
        self.val_data=val_dataset
        self.args=args
        self.model=model
        self.device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.optimizer=AdamW(self.model.parameters(),args.lr,weight_decay=args.weight_decay)
    def train_model(self):
        save_model_dir=os.path.join(self.args.saved_dir,self.args.train_model_dir)#save model dir

        #build data loader
        sampler=RandomSampler(self.train_data)
        data_iter=DataLoader(self.train_data,batch_size=self.args.train_bs,
                             sampler=sampler,collate_fn=create_mini_batch,pin_memory=True)
        

        if self.args.epochs>0:
            total_steps=self.args.epochs*(len(data_iter)//self.args.grad_accumulate_steps)
            num_epochs=self.args.epochs
        elif self.args.total_steps>0:
            num_epochs=self.args.total_steps//(len(data_iter)//self.args.grad_accumulate_steps)
            total_steps=self.args.total_steps
        
        #build lr rate scheduler
        lr_scheduler=get_linear_schedule_with_warmup(self.optimizer,self.args.warm_steps,total_steps)
        #build grad scaler for amp operations
        scaler=torch.cuda.amp.GradScaler()
        #build train progress bar
        train_pgb=trange(num_epochs,desc='EPOCHS')
        global_steps=0
        global_loss=0
        global_predicts=None
        global_labels=None

        #train!
        logger.info('****Start to Training!****')
        logger.info(f'Train example nums:{len(self.train_data)}')
        logger.info(f'Batch size:{self.args.train_bs}')
        logger.info(f'Epochs:{num_epochs}')
        logger.info(f'Pos class weights: {self.args.pos_weights}')
        logger.info(f'trainable step:{total_steps}')
        logger.info(f'Current lr:{self.args.lr}')
        logger.info(f'lr warm steps:{self.args.warm_steps}')
        logger.info(f'gradient accumulate steps:{self.args.grad_accumulate_steps}')
        logger.info(f'logging steps:{self.args.logging_steps}')
        logger.info(f'save steps:{self.args.save_steps}')
        self.model.zero_grad()
        for ep in train_pgb:
            epochs_pgb=tqdm(data_iter,desc='iterations')

            for batch in epochs_pgb:
                self.model.train()
                global_steps+=1
                if next(self.model.parameters()).is_cuda:
                    inputs={}
                    for f_name,data in batch.items():
                        if f_name=='title':
                            inputs[f_name]=[t.long().to(self.device) for t in data]
                        else:
                            inputs[f_name]=data.to(self.device)
                with torch.cuda.amp.autocast():
                    #forward pass
                    outputs,loss=self.model(**inputs)
                    
                #scaling loss & compute gradient
                global_loss+=loss.item()
                scaler.scale(loss).backward()                

                #update model weights
                if global_steps%self.args.grad_accumulate_steps==0:

                    #unscaled grad for grad clipping
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_norm)

                    scaler.step(self.optimizer)#update unscaled grad
                    scaler.update()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()#clear leaf variable grad

                #get predict & labels
                #outputs=[Bs,1]
                #predicts=[Bs,]
                if global_predicts is None:
                    global_predicts=torch.sigmoid(outputs.reshape(-1))
                    global_labels=inputs['label']
                else:
                    outputs=torch.sigmoid(outputs.reshape(-1))
                    global_predicts=torch.cat((global_predicts,outputs),0)
                    global_labels=torch.cat((global_labels,inputs['label']),0)
                
                #evaluate model
                if (global_steps%self.args.logging_steps==0) and (self.args.logging_steps>0):
                    #get predict label
                    predicts=torch.where(global_predicts>self.args.prob,1,0).cpu().numpy()
                    labels=global_labels.cpu().numpy()
                    #evalute model with train data
                    train_metrics=get_metrics(predicts,labels)
                    #logging train info
                    logger.info(f'****Model train merics for each {self.args.logging_steps}****')
                    logger.info(f'Model global loss:{global_loss/self.args.logging_steps}')
                    for n,s in train_metrics.items():
                        logger.info(f'{n} : {s}')

                    self.eval_model(self.val_data)

                    #clear metrics for each logging steps
                    global_labels=None
                    global_predicts=None
                    global_loss=0
                    
                #save model state for each step ranges
                if (global_steps%self.args.save_steps==0) and (self.args.save_steps>0):
                    self.save_model(self.model,save_model_dir)
                
                if 0<(total_steps*self.args.grad_accumulate_steps)<global_steps:
                    epochs_pgb.close()
                    break
            if 0<(total_steps*self.args.grad_accumulate_steps)<global_steps:
                train_pgb.close()
                break
        self.save_model(self.model,save_model_dir)
    def eval_model(self,eval_data):
        #build eval data loader
        sampler=SequentialSampler(eval_data)
        data_iter=DataLoader(eval_data,batch_size=self.args.val_bs,sampler=sampler,
                            collate_fn=create_mini_batch,pin_memory=True)
        preds=None
        labels=None
        total_loss=0
        self.model.eval()
        for batch in data_iter:
            with torch.no_grad():
                if next(self.model.parameters()).is_cuda:
                    inputs={}
                    for f_name,data in batch.items():
                        if f_name=='title':
                            inputs[f_name]=[t.long().to(self.device) for t in data]
                        else:
                            inputs[f_name]=data.to(self.device)
                
                #predict data
                #outputs=[Bs,1]
                outputs,loss=self.model(**inputs)
                total_loss+=loss.item()

                #get predicts & labels
                if preds is not None:
                    preds=torch.cat((preds,torch.sigmoid(outputs.reshape(-1))),0)
                else:
                    preds=torch.sigmoid(outputs.reshape(-1))
                if labels is not None:
                    labels=torch.cat((labels,inputs['label']),0)
                else:
                    labels=inputs['label']
        
        #get label
        preds=torch.where(preds>self.args.prob,1,0).cpu().numpy()
        labels=labels.cpu().numpy()
        #start to eval model
        eval_metrics=get_metrics(preds,labels)

        logger.info('****Start to evaluate Model****')
        logger.info(f'Data num:{len(eval_data)}')
        logger.info(f'Batch size:{self.args.val_bs}')
        logger.info(f'eval_loss : {total_loss/len(data_iter)}')

        for n,s in eval_metrics.items():
            logger.info(f'{n} : {s}')

    def save_model(self,model,save_model_dir):
        #create model dir
        if os.path.isdir(save_model_dir):
            logger.info(f'Model dir: {save_model_dir} Already exist!')
        else:
            logger.info(f'Model dir: {save_model_dir} is not exist!')
            os.makedirs(save_model_dir)
            logger.info('Create saved model success!')
        
        try:
            if isinstance(model,PreTrainedModel):
                model.save_pretrained(save_model_dir)
            else:
                model_state=model.state_dict()
                torch.save(model_state,os.path.join(save_model_dir,'model_state.pt'))
            logger.info(f'Save model state to {save_model_dir} success!')
        except:
            logger.info('Save model failed... some things is wrong')

    @classmethod
    def load_model(cls,model,train_dataset,val_dataset,args):
        #build save model dir
        save_model_dir=os.path.join(args.saved_dir,args.train_model_dir)
        #check model dir is exist or not
        if not os.path.isdir(save_model_dir):
            raise FileNotFoundError("Saved model folder don't exists")
        try:
            if issubclass(model,PreTrainedModel):
                pretrain_model=model.from_pretrained(save_model_dir)
                logger.info(f'Loading pretrained model from {save_model_dir} success!')
        except:
            if isinstance(model,torch.nn.Module):
                model_state=torch.load(save_model_dir)
                pretrain_model=model.load_state_dict(model_state)
                logger.info(f'Loading pretrained model from {save_model_dir} success!')
            else:
                raise Exception('Load class of model incorrect!')

        return cls(train_dataset,val_dataset,pretrain_model,args)
        

        





