import torch
from torch import nn
from torch.nn import functional as F
from utils_model import fake_classifier,item_extractor,Attentioner
import numpy as np
import os

class RnnExtractor(nn.Module):
    def __init__(self,input_embed,embed_dim,hid_dim,n_layers,
                using_pretrain_weight=True,padding_idx=0,dropout_rate=0.1):

        super(RnnExtractor,self).__init__()

        #determined using pretrain weight
        if using_pretrain_weight:
            text_embed=np.load(r'.\Data\fakeJob\vocab_embed\fastText_300d_25502_embed.npy')
            text_embed=torch.from_numpy(text_embed)
            self.embed_layer=nn.Embedding.from_pretrained(text_embed,freeze=False,padding_idx=padding_idx)
        else:
            self.embed_layer=nn.Embedding(input_embed,embed_dim,padding_idx=padding_idx)
        
        self.rnn_layer=nn.GRU(embed_dim,hid_dim,num_layers=n_layers,batch_first=True,dropout=dropout_rate)

    def forward(self,input_tensors):

        hiddens=self.embed_layer(input_tensors).double()
        outputs,_=self.rnn_layer(hiddens.float())
        return outputs

class RnnFakeDetector(nn.Module):
    def __init__(self,text_vocab,item_vocab,embed_dim,hid_dim,fc_dim,meta_dim,output_dim,rnn_layerN,
                     fc_layerN,padding_idx=0,dropout_rate=0.1,bidirectional=False,using_pretrain_weight=True):

        super(RnnFakeDetector,self).__init__()

        #build embed weight
        self.item_embed=item_extractor(item_vocab,embed_dim,padding_idx)
        #build rnn extractors
        self.rnn_extractors=nn.ModuleList([RnnExtractor(text_vocab,embed_dim,hid_dim,rnn_layerN,using_pretrain_weight)
                                           for _ in range(4)])
        #build concat layer
        self.cp_cat=nn.Linear(hid_dim*2,hid_dim)
        self.job_cat=nn.Linear(hid_dim*2,hid_dim)
        self.text_concat=nn.Linear(hid_dim+(embed_dim*2),hid_dim)
        self.item_attner=Attentioner()
        self.hid2Embed=nn.Linear(hid_dim,embed_dim)
        #build downstream model
        self.classifier=fake_classifier(meta_dim+(hid_dim*2),fc_dim,output_dim,fc_layerN)

        #build loss & activ.
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()

        self.criterion=nn.BCEWithLogitsLoss()
    
    def forward(self,cp_file=None,desc=None,require=None,benefits=None,
                title=None,meta_data=None,labels=None):

        contexts=[]
        texts=[cp_file,benefits,desc,require]

        for  rnn,text in zip(self.rnn_extractors,texts):
            context=rnn(text)#using rnn to capture context
            contexts.append(context)
        
        #concat & transform each text
        cp_context=torch.hstack([torch.mean(t,1) for t in contexts[:2]])
        job_context=torch.hstack([torch.mean(t,1) for t in contexts[2:]])

        cp_context=self.tanh(self.cp_cat(cp_context))
        job_context=self.tanh(self.job_cat(job_context))

        #get_item embed=[Bs,embed_dim]
        item_contexts=self.item_embed(title)

        #convert query context
        #desc_context=[Bs,seqLen,embed_dim]
        desc_context=self.hid2Embed(contexts[2])

        #compute attention for item
        #[Bs,embed_dim]
        item_attned=self.item_attner(item_contexts,desc_context.transpose(1,2),desc_context)

        #concat item & job_context
        #job_context=[Bs,hid_dim]
        job_context=self.text_concat(torch.cat((item_contexts,item_attned,job_context),1))

        #feed job_context with item,meta_data,cp_context to downstream model
        #output logitics=[Bs,1]
        output_logitics=self.classifier(torch.cat((job_context,cp_context,meta_data),1))

        loss=None
        if labels is not None:
            loss=self.criterion(output_logitics.reshape(-1),labels)
        
        return output_logitics,loss






        