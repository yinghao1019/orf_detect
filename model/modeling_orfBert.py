import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel,BertPreTrainedModel,BertTokenizer,BertConfig
from .utils_model import fake_classifier,item_extractor,Attentioner

class BertFakeDetector(BertPreTrainedModel):
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
        
        def __init__(self,config,hid_dim,item_input_dim,embed_dim,fc_dim,
                     meta_dim,output_dim,pos_weight,maxLen,padding_idx=0,
                     vocab_num,using_pretrain_weight=True,dropout_rate=0.2):
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

                super(BertFakeDetector,self).__init__(config)
                
                #model attrs
                self.bert_hid=config.hidden_size
                self.hid_dim=hid_dim
                self.embed_dim=embed_dim
                self.fc_dim=fc_dim
                self.meta_dim=meta_dim
                self.ouput_dim=output_dim
                self.dropout_rate=dropout_rate
                self.config=config
                
                #build model
                self.bert1=BertModel(config)
                self.bert2=BertModel(config)
                self.job_rnn=nn.GRU(self.bert_hid,hid_dim,num_layers=1,
                                    batch_first=True,dropout=dropout_rate)
                self.cp_rnn=nn.GRU(self.bert_hid,hid_dim,num_layers=1,
                                    batch_first=True,dropout=dropout_rate)
                self.item_model=item_extractor(item_input_dim,embed_dim,padding_idx=padding_idx,
                                               using_pretrain_weight=using_pretrain_weight)
                
                #build concat layer
                self.rnn_cat=nn.Linear(self.bert_hid*2,hid_dim)
                self.meta_cat=nn.Linear(embed_dim+meta_dim,hid_dim)
                #build fake classifier
                self.fake_detector=fake_classifier(hid_dim*3,hid_dim,output_dim,2)
                self.attner=Attentioner(embed_dim,self.bert_hid,self.bert_hid,hid_dim)
                #build other func
                self.tanh=nn.Tanh()
                self.dropout=nn.Dropout(dropout_rate)
                self.leakyRelu=nn.LeakyReLU()
                self.bn=nn.BatchNorm1d(embed_dim+meta_dim)
                self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

                self.bert1.resize_token_embeddings(vocab_num)
                self.bert2.resize_token_embeddings(vocab_num)
        def forward(self,job_tokens=None,cp_tokens=None,job_segs=None,
                cp_segs=None,job_masks=None,cp_masks=None,
                title=None,meta_data=None,label=None,
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
                job_outputs=self.bert(input_ids=job_tokens,attention_mask=job_masks,
                                      token_type_ids=job_segs,output_hidden_states=output_hidden_states,
                                      return_dict=return_dict)
                cp_outputs=self.bert(input_ids=cp_tokens,attention_mask=cp_masks,
                                     token_type_ids=cp_segs,output_hidden_states=output_hidden_states,
                                     return_dict=return_dict)
                #job_hidden=[Bs,seqLen,hid_dim]
                #cp_hidden=[Bs,seqLen,hid_dim]
                job_hiddens=job_outputs[0]
                cp_hiddens=cp_outputs[0]

                #extract items embed for title
                #item_embed=[Bs,embed_dim]
                item_embed=self.dropout(self.item_model(title).float())

                #item attend to job contexts
                item_attend=self.attner(item_embed,job_hiddens[:,1:,:],job_hiddens[:,1:,:])

                #using mean to extract context
                #job_hiddens=[Bs,seqLen,hid]
                #cp_hiddens=[Bs,seqLen,hid]
                job_hiddens=job_hiddens.mean(dim=1)
                cp_hiddens=cp_hiddens.mean(dim=1)

                #concat context & transform
                #context_outputs=[Bs,hid_dim]
                context_outputs=self.rnn_cat(torch.cat((job_hiddens,cp_hiddens),1))
                # print('context outs:{}'.format(context_outputs))
                
                context_outputs=self.leakyRelu(context_outputs)
                #print('context outs1:{}'.format(context_outputs))
                
                # print('item_embed outs:{}'.format(item_embed))
                #normalization meta data
                meta_hiddens=self.bn(torch.cat((item_embed,meta_data),1))
                meta_hiddens=self.dropout(self.leakyRelu(self.meta_cat(meta_hiddens)))
                # print('meta hid outs:{}'.format(meta_hiddens))
                # print('meta hid outs1:{}'.format(meta_hiddens))
                 
                #output_logitics=[Bs,1]
                hiddens=torch.cat((context_outputs,meta_hiddens,item_attend),1)
                # print('Before final output:{}'.format(hiddens))
                output_logitics=self.fake_detector(hiddens)
                # print('final output:{}'.format(output_logitics))
                loss=None
                if label is not None:
                        loss=self.criterion(output_logitics,label.float())
                
                return output_logitics,loss



