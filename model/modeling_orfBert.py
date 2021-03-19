import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel,BertPreTrainedModel,BertTokenizer,BertConfig
from .utils_model import fake_classifier,item_extractor


class BertFakeDetector(BertPreTrainedModel):
    def __init__(self,config,hid_dim,output_dim,embed_dim,fc_dim,
                      meta_dim,fc_layerN,vocab_num,pos_weight,
                      padding_idx=0,using_pretrain_weight=True,dropout_rate=0.1):

        super(BertFakeDetector,self).__init__(config)

        self.config=config
        #build bert
        self.bert=BertModel(config)
        #using gru to extract 
        self.job_rnn=nn.GRU(config.hidden_size,hid_dim,batch_first=True)
        self.cp_rnn=nn.GRU(config.hidden_size,hid_dim,batch_first=True)
        #build embed
        self.item_embed=item_extractor(vocab_num,embed_dim,padding_idx=padding_idx,
                                       using_pretrain_weight=using_pretrain_weight)

        #build concat layer
        self.rnn_cat=nn.Linear(hid_dim*2,hid_dim)
        self.meta_cat=nn.Linear(embed_dim+meta_dim,hid_dim)

        #build downstream model
        self.classifier=fake_classifier(hid_dim+embed_dim,hid_dim,output_dim,fc_layerN)

        #build other func
        self.tanh=nn.Tanh()
        self.leakyRelu=nn.LeakyReLU()
        self.bn2=nn.BatchNorm1d(embed_dim+meta_dim)
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    
    def forward(self,job_tokens=None,cp_tokens=None,job_segs=None,
                cp_segs=None,job_mask=None,cp_mask=None,
                title=None,meta_data=None,label=None,
                output_hidden_states=False,return_dict=None):

        return_dict=return_dict if return_dict is not None else self.config.use_return_dict

        #extract job & cp context
        job_outputs=self.bert(input_ids=job_tokens,attention_mask=job_mask,
                token_type_ids=job_segs,output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        cp_outputs=self.bert(input_ids=cp_tokens,attention_mask=cp_mask,
                token_type_ids=cp_segs,output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        
        #job_hidden=[Bs,seqLen,hid_dim]
        #cp_hidden=[Bs,seqLen,hid_dim]
        job_hiddens=job_outputs[0]
        cp_hiddens=cp_outputs[0]

        #using rnn strategy to extract context
        #job_hiddens=[Bs,seqlen,hid_dim]
        #cp_hiddens=[Bs,seqlen,hid_dim]
        job_hiddens,_=self.job_rnn(job_hiddens[:,1:,:])
        cp_hiddens,_=self.cp_rnn(cp_hiddens[:,1:,:])
        
        #concat context & transform
        #context_outputs=[Bs,hid_dim]
        context_outputs=self.rnn_cat(torch.cat((job_hiddens[:,-1,:],cp_hiddens[:,-1,:]),1))
        # print('context outs:{}'.format(context_outputs))
        context_outputs=self.leakyRelu(context_outputs)
        # print('context outs1:{}'.format(context_outputs))
        #extract items embed for title
        #item_embed=[Bs,embed_dim]
        item_embed=self.item_embed(title).float()
        # print('item_embed outs:{}'.format(item_embed))
        #normalization meta data
        # meta_hiddens=self.meta_cat()
        # print('meta hid outs:{}'.format(meta_hiddens))
        # meta_hiddens=self.tanh(meta_hiddens)
        # print('meta hid outs1:{}'.format(meta_hiddens))

        #output_logitics=[Bs,1]
        hiddens=torch.cat((context_outputs,item_embed),1)
        # print('Before final output:{}'.format(hiddens))
        output_logitics=self.classifier(hiddens)
        # print('final output:{}'.format(output_logitics))
        loss=0
        if label is not None:
                loss=self.criterion(output_logitics,label.unsqueeze(1).float())


        return output_logitics,loss



