import torch
from torch import nn
from torch.nn import functional as F
from .utils_model import fake_classifier,item_extractor,Attentioner
import numpy as np
import os


class RnnExtractor(nn.Module):
    """
    A class to build rnn_layer with embedding for fakeJobdetection.
    The class is inheriented from torch Module.

    ...

    Attributes
    ----------
    input_embed : int
        Set the vocab nums that you want to convert to embed vector.
    embed_dim : int
        Set the dim for embed_vector.
    hid_dim : int
        Set the rnn neuron nums.
    n_layers : int
        Set the rnn_layer nums.

    Methods
    -------
    forward(input_tensors):
       compute input_token to extract useful knowledge.
    """
    def __init__(self,input_embed,embed_dim,hid_dim,n_layers,
                using_pretrain_weight=True,padding_idx=0,dropout_rate=0.1):
        """
        Set the model params to construct architecture for the text rnn extractor.

        Parameters
        ----------
            input_embed : int
                Set the dim for input_layer.Which dim equal to vocab num
            embed_dim : int
                Set the dim for embed_vector.
            hid_dim : int
                Set the dim for rnn model.
            n_layers : int
                Set layer num to use stack rnn.
            using_pretrain_weight : int ,optional
                Whether or not to use pretrain weight for embed_layer.
            padding_idx : int ,optional
                Set indice for pad token to build zero embed vector.
            dropout_rate : float ,optional
                Set the regularzation to rnn model.
        """
        super(RnnExtractor,self).__init__()
        #set model attr
        self.input_embed=input_embed
        self.embed_dim=embed_dim
        self.hid_dim=hid_dim
        self.n_layers=n_layers

        #determined using pretrain weight or not for job texts
        if using_pretrain_weight:
            text_embed=np.load(r'./Data/fakeJob/vocab_embed/orig_cbw_128d_32145_embed.npy')
            text_embed=torch.from_numpy(text_embed)
            self.embed_layer=nn.Embedding.from_pretrained(text_embed,freeze=False,padding_idx=padding_idx)
        else:
            self.embed_layer=nn.Embedding(input_embed,embed_dim,padding_idx=padding_idx)
        
        self.rnn_layer=nn.GRU(embed_dim,hid_dim,num_layers=n_layers,batch_first=True,dropout=dropout_rate)
    def forward(self,input_tensors):
        """
        Extract input_tensors hidden state using build model.
        which hidden state will be vector of sequence.
        Parameters
        ----------
        input_tensors : torch.LongTensor 
            The input tensor is extracted hidden state(shape=[Bs,seqLen,hid_dim]).

        Returns
        -------
        output_hiddens : torch.FloatTensor

        """
        hiddens=self.embed_layer(input_tensors)
        hid_outputs,_=self.rnn_layer(hiddens.float())
        
        return hid_outputs

class RnnFakeDetector(nn.Module):
    """
    A class to build GRU-DNN  model for fakeJobdetection.
    The class is inheriented from torch Module.

    ...

    Attributes
    ----------
    text_input_dim : int
        Set the text vocab nums that you want to convert to embed vector.
    item_input_dim : int
        Set the item vocab nums that you want to convert to embed vector.
    embed_dim : int
        The embed_dim for item & job text.
    hid_dim : int
        The hid_dim for rnn,concat layer
    fc_dim : int
        The fc_dim for fake model's hid_layer.
    meta_dim : int
        The dim for meta data's num.
    output_dim : int
        The output_class num.
    dropout_rate : float
        The dropout rate for all model regularization.

    Methods
    -------
    forward(cp_file,desc,require,benefits,
            title,meta_data,labels):
       Compute fake prob for given texts.If provide
       true label.It's will compute loss.
    """
    def __init__(self,text_input_dim,item_input_dim,embed_dim,hid_dim,
                 fc_dim,meta_dim,output_dim,pos_weight,padding_idx=0,
                 dropout_rate=0.1,bidirectional=False,using_pretrain_weight=False):
        """The require args for build model.

        Parameters
        ----------
            text_input_dim : int
                Set the text vocab nums that you want to convert to embed vector.
            item_input_dim : int
                Set the item vocab nums that you want to convert to embed vector.
            embed_dim : int
                The embed_dim for item & job text.
            hid_dim : int
                The hid_dim for rnn,concat layer
            fc_dim : int
                The fc_dim for fake model's hid_layer.
            meta_dim : int
                The dim for meta data's num.
            output_dim : int
                The output_class num.
            pos_weight : int
                The positive weight for balanced class propotion.
            padding_idx : int , optional
                Set indice for pad token to build zero embed vector.
            using_pretrain_weight : True ,optional
                Whether  or not to use pretrain embed weight.
            dropout_rate : float ,optional
                The dropout rate for all model regularization.

        """
        super(RnnFakeDetector,self).__init__()

        #build model
        self.item_embed=item_extractor(item_input_dim,embed_dim,padding_idx,
                                       using_pretrain_weight=using_pretrain_weight)
        self.rnn_extractors=nn.ModuleList([RnnExtractor(text_input_dim,embed_dim,hid_dim,2,using_pretrain_weight,
                                           dropout_rate=dropout_rate) for _ in range(4)])
        
        self.cp_cat=nn.Linear(hid_dim*2,hid_dim)
        self.job_cat=nn.Linear(hid_dim*2,hid_dim)
        self.text_concat=nn.Linear(hid_dim*3,hid_dim)#job+title text 
        self.embed2hid=nn.Linear(embed_dim,hid_dim)
        self.item_attner=Attentioner(hid_dim,hid_dim,hid_dim,hid_dim)#item attnetion
        self.bn=nn.BatchNorm1d(meta_dim)
        #build downstream model
        self.classifier=fake_classifier(hid_dim*2,fc_dim,output_dim,3)

        #build loss & activation.
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    
    def forward(self,cp_file=None,desc=None,require=None,benefits=None,
                title=None,meta_data=None,label=None):
        """
        Compute the fake prob for given text & data.

        If label is providied,then it will compute given data's loss with true label,
        if not,The loss will be None.

        Parameters
        ----------
        cp_file : batch of torch.LongTensor.
               Indices of company_profile sequence tokens in the vocabulary. 
        desc : batch of torch.LongTensor.
               Indices of description sequence tokens in the vocabulary.
        require : batch of torch.LongTensor.
               Indices of requires sequence tokens in the vocabulary.
        benefits : batch of torch.LongTensor.
               Indices of benefits sequence tokens in the vocabulary.
        title : batch of torch.LongTensor.
               Indices of title sequence tokens in the vocabulary.
        meta_data : batch of torch.LongTensor.
               The numerics for data engineering features.
        labels : batch of torch.LongTensor.
               Class indexs for each seq.
                       
        Returns
        -------
        predict_logitics,loss
        """
        outputs=[]
        texts=[cp_file,benefits,desc,require]

        for  rnn,text in zip(self.rnn_extractors,texts):
            output=rnn(text.long())#using rnn to capture context
            outputs.append(output)
        
        #concat & transform each text
        #use mean pooling for each context
        cp_context=torch.hstack([t.mean(dim=1) for t in outputs[:2]])
        job_context=torch.hstack([t.mean(dim=1) for t in outputs[2:]])

        cp_context=self.tanh(self.cp_cat(cp_context))
        job_context=self.tanh(self.job_cat(job_context))

        #get_item embed=[Bs,embed_dim]
        item_contexts=self.item_embed(title)

        #convert query context
        #desc_context=[Bs,seqLen,embed_dim]
        item_contexts=self.embed2hid(item_contexts.float())
        desc_contexts=outputs[2]
        #compute attention for item
        #[Bs,embed_dim]
        item_attned=self.item_attner(item_contexts,desc_contexts,desc_contexts)

        #concat item & job_context
        #job_context=[Bs,hid_dim]
        job_context=self.text_concat(torch.cat((item_contexts,item_attned,job_context),1))

        #feed job_context with item,meta_data,cp_context to downstream model
        #output logitics=[Bs,1]
        output_logitics=self.classifier(torch.cat((job_context,cp_context),1))

        loss=None
        if label is not None:
            loss=self.criterion(output_logitics,label.float())
        
        return output_logitics,loss






        