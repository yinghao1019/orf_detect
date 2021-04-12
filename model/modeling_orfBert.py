import torch
import os
import logging
from torch import nn
from torch.nn import functional as F
from transformers import BertModel,BertPreTrainedModel,BertTokenizer,BertConfig
from .utils_model import fake_classifier,item_extractor,Attentioner
logger=logging.getLogger(__name__)


class Bert_layer(BertPreTrainedModel):
        """
        A class to build deep learning model based on BERT
        for Fake job detection.The class is inheriented 
        from BertPreTrainedModel of transformers package.
        Because we using pretrain bert to fine-tuning model,
        so you must using from_pretrained to loading pretrain
        weight.See more information from BertPretrained class.
        ...
        
        Attributes
        ----------
        bert_hid : int
           The bert model output hid dim.
        hid_dim : int
           The hid_dim for rnn,concat layer
        embed_dim : int
           The embed_dim for item text.
        fc_dim : int
           The fc_dim for fake model's hid_layer.
        meta_dim : int
           The dim for meta data's num.
        output_dim : int
           The output_class num.
        dropout_rate : float
           The dropout rate for regularization.

        Methods
        -------
        foward(job_tokens,cp_tokens,job_segs=None,
               cp_segs=None,job_mask=None,cp_mask=None,
               title=None,meta_data=None,label=None,
               output_hidden_states=False,return_dict=None):

               Compute fake prob for given texts.If provide
               True label.It's will compute loss.
        """
        
        def __init__(self,config,hid_dim):
                """The require args for build model.

                Parameters
                ----------
                    config : PretrainedConfig object
                        The pretrained Bert configuration.
                    bert_hid : int
                        The bert model output hid dim.
                    hid_dim : int
                        The hid_dim for rnn,concat layer
                    item_embed : int
                        The embed_dim for item text.
                    fc_dim : int
                        The fc_dim for fake model's hid_layer.
                    meta_dim : int
                        The dim for meta data's num.
                    output_dim : int
                        The output_class num.
                    pos_weight : int
                       The positive weight for balanced class propotion.
                    padding_idx : int
                       The pad token's index for embed layer.
                    using_pretrain_weight : True
                       Using pretrain embed model or not.
                    dropout_rate : float
                        The dropout rate for regularization.
                """

                super(Bert_layer,self).__init__(config)
                
                #model attrs
                self.bert_hid=config.hidden_size
                self.hid_dim=hid_dim
                self.config=config
                
                #build model
                self.bert=BertModel(config)
                #build concat layer
                self.ffn=nn.Linear(self.bert_hid,hid_dim,bias=True)
                self.tanh=nn.Tanh()
        def forward(self,token_ids=None,seg_ids=None,mask_ids=None,
                   output_hidden_states=False,return_dict=None):
                """
                Compute the fake prob for given text & data.

                If label is providied,then it will compute given data's loss with true label,
                if not,The loss will be None.

                Parameters
                ----------
                    job_tokens : batch of torch.LongTensor.
                        Indices of job sequence tokens in the vocabulary. 
                    cp_tokens : batch of torch.LongTensor.
                        Indices of cp sequence tokens in the vocabulary.
                    job_segs : batch of torch.LongTensor.
                        Segment token indices to indicate first and second portions of the jobs.
                    cp_segs : batch of torch.LongTensor.
                        Segment token indices to indicate first and second portions of the cps.
                    job_masks : batch of torch.LongTensor.
                        Mask to avoid performing attention on padding job_token indices. 
                    cp_masks : batch of torch.LongTensor.
                        Mask to avoid performing attention on padding cp_token indices. 
                    title : batch of torch.LongTensor.
                        Indices of title sequence tokens in the vocabulary.
                    meta_data : batch of torch.FloatTensor.
                        The numerics for data engineering features.
                    label : batch of torch.LongTensor.
                        Class indexs for each seq.
                    output_hid_states : Boolean
                        Whether or not to return the hidden states of all layers.
                    return_dict : Boolean
                       Whether or not to return a ModelOutput instead of a plain tuple.
                       
                Returns
                -------
                predict_logitics,loss
                """

                
                return_dict=return_dict if return_dict is not None else self.config.use_return_dict
                
                #extract job & cp context
                outputs=self.bert(input_ids=token_ids,attention_mask=mask_ids,
                                      token_type_ids=seg_ids,output_hidden_states=output_hidden_states,
                                      return_dict=return_dict)
                #hiddens=[Bs,seqLen,bert_hid]
                hiddens=outputs[0]
                #non-linear transform
                #hiddens=[Bs,seqLen,hid_dim]
                hiddens=self.ffn(hiddens)

                return self.tanh(hiddens)

class Bert_Fakedetector(nn.Module):
    def __init__(self,pretrain_path1,pretrain_path2,vocab_num,item_input_dim,
                     hid_dim,embed_dim,fc_dim,meta_dim,output_dim,pos_weight,
                     padding_idx=0,using_pretrain_weight=True,dropout_rate=0.2):
        super(Bert_Fakedetector,self).__init__()

        #build pretrain bert layer
        self.job_bert=Bert_layer.from_pretrained(pretrain_path1,hid_dim=hid_dim)
        self.cp_bert=Bert_layer.from_pretrained(pretrain_path2,hid_dim=hid_dim)
        self.job_bert.resize_token_embeddings(vocab_num)
        self.cp_bert.resize_token_embeddings(vocab_num)
        #build other model
        self.item_model=item_extractor(item_input_dim,embed_dim,padding_idx=padding_idx,
                                       using_pretrain_weight=using_pretrain_weight)
        self.item_attn_layer=Attentioner(embed_dim,hid_dim,hid_dim,hid_dim)
        self.context_layer=nn.Linear(hid_dim*2,hid_dim)
        self.item_cat=nn.Linear(embed_dim+hid_dim,hid_dim)
        self.fake_model=fake_classifier(hid_dim*2+meta_dim,fc_dim,output_dim,2)

        #build other criterion
        self.tanh=nn.Tanh()
        self.dropout=nn.Dropout(dropout_rate)
        self.leakyRelu=nn.LeakyReLU()
        self.bn=nn.BatchNorm1d(meta_dim)
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self,job_tokens=None,cp_tokens=None,job_segs=None,
                cp_segs=None,job_masks=None,cp_masks=None,
                title=None,meta_data=None,label=None,):
        
        #xx_tokens,xx_seqgs,xx_masks=[Bs,seqlen]
        #title=[Bs,seqLen]
        #meta_data=[Bs,meta_dim]
        #label=[Bs,class]

        #get bert embed 
        #xx_contexts=[Bs,hid_dim],xx_hiddens=[Bs,seqLen,bert_hid]
        job_hiddens=self.job_bert(token_ids=job_tokens,mask_ids=job_masks,
                                 seg_ids=job_segs)
        cp_hiddens=self.cp_bert(token_ids=cp_tokens,mask_ids=cp_masks,
                                seg_ids=cp_segs)
        
        #compute item attention with job
        #items=[Bs,embed_dim]
        #item_attened=[Bs,hid_dim]
        items=self.item_model(title)
        item_attened=self.item_attn_layer(items,job_hiddens,job_hiddens)

        #concat job & cp layer
        #contexts=[Bs,hid_dim]
        contexts=self.context_layer(torch.cat((job_hiddens.mean(dim=1),cp_hiddens.mean(dim=1)),1))
        contexts=self.dropout(self.leakyRelu(contexts))

        #meta_hiddens=[Bs,hid_dim]
        item_hiddens=torch.cat((items,item_attened),dim=1)
        item_hiddens=self.item_cat(item_hiddens)

        #feed all hiddens to fake classifier
        #hiddens=[Bs,hid_dim*3]
        #output_logitics=[Bs,1]
        hiddens=torch.cat((contexts,item_hiddens,self.bn(meta_data)),1)
        output_logitics=self.fake_model(hiddens)

        #compute loss
        loss=None
        if label is not None:
            loss=self.criterion(output_logitics,label.float())
        
        return output_logitics,loss

    def save_pretrained(self,save_model_dir):
        #build save xxbert dir
        cp_bdir=os.path.join(save_model_dir,'cp_bert')
        job_bdir=os.path.join(save_model_dir,'job_bert')
        #check whether sub dir exist or not
        if os.path.isdir(job_bdir):
            logger.info('Sub directory for save bert state is exists!!')
        else:
            logger.info('Sub directory for save bert state not exists!So create new')
            os.makedirs(cp_bdir)
            os.makedirs(job_bdir)
            logger.info(f'Create new dir for {job_bdir} & {cp_bdir}')
        try:
            self.job_bert.save_pretrained(job_bdir)
            self.cp_bert.save_pretrained(cp_bdir)
            logger.info(f'Save bert model state success!')
        except:
            logger.info(f'Save bert model state succes Failed!')




