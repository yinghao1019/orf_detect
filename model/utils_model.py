import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
#Universal model params config
model_uniConfig={"embed_dim":128,'hid_dim':256,"fc_dim":256,"meta_dim":13,
              "output_dim":1,"dropout_rate":0.3,"using_pretrain_weight":True,
              }

class fake_classifier(nn.Module):
    """
    The model for output fake prob.Which is inherited from torch Module.

    ...

    Attributes
    ----------
    input_dim : int
        Set the dim for input_layer.
    inter_dim : int
        Set the dim for hid_layer.
    output_dim : int
        Set the dim for output_layer.

    Methods
    -------
    foward(input_tensors):
        The callable method will compute class logitics for given input_tensors.
        The class num equal to output_dim.
    """

    def __init__(self,input_dim,inter_dim,output_dim,n_layers):
        """
        Constructs architecture for the classifier.

        Parameters
        ----------
            input_dim : int
                Set the dim for input_layer.
            inter_dim : int
                Set the dim for hid_layer.
            output_dim : int
                Set the dim for output_layer.
            n_layers : int
                Determined the hid_layer num.
        """
        super(fake_classifier,self).__init__()
        #build module dim attr
        self.input_dim=input_dim
        self.inter_dim=inter_dim
        self.output_dim=output_dim

        #build model
        self.trans_layer=nn.Linear(input_dim,inter_dim)
        self.inter_layer=nn.ModuleList([nn.Linear(inter_dim//pow(2,i),inter_dim//pow(2,i+1)) 
                                        for i in range(n_layers)])
        #build output layer
        self.output_layer=nn.Linear(inter_dim//pow(2,n_layers),output_dim)

        #build activation func
        self.leakyRelu=nn.LeakyReLU()
    
    def forward(self,input_tensors):
        """
        Compute class logitics according output_dim for given tensors

        Parameters
        ----------
        input_tensors : torch.tensor, optional
            The tensors that you want to predict

        Returns
        -------
        logitics :torch.tensor
        """
        hiddens=self.leakyRelu(self.trans_layer(input_tensors))
        for l in self.inter_layer:
            hiddens=self.leakyRelu(l(hiddens))
        return self.output_layer(hiddens)

class item_extractor(nn.Module):
    """
    The model used for loading item text embed,and it's will use
    pooling strategy to combine token embeds of sequence.

    ...

    Attributes
    ----------
    input_embed : int
        Set the vocab nums that you want to convert to embed vector.
    embed_dim : int
        Set the dim for hid_layer.
    padding_idx : int
        Set indice for pad token to build zero embed vector.
    using_pretrain_weight:bool
        Whether or not to use pretrained weight to set init embed weight

    Methods
    -------
    foward(input_tensors):
        The callable method will compute class logitics for given input_tensors.
        The class num equal to output_dim.
    """
    def __init__(self,input_embed,embed_dim,padding_idx,using_pretrain_weight=False):
        """
        Constructs 

        Parameters
        ----------
            input_embed : int
                Set the vocab nums that you want to convert to embed vector.
            embed_dim : int
                Set the dim for embed_vector.
            padding_idx : int
                Set indice for pad token to build zero embed vector.
            using_pretrain_weight : bool
                Whether or not to use pretrained weight to set init embed weight.
        """

        super(item_extractor,self).__init__()
        #set attr
        self.input_embed=input_embed
        self.embed_dim=embed_dim
        self.using_pretrain_weight=using_pretrain_weight

        #build embed layer
        if using_pretrain_weight:
            item_embeds=np.load(r'./Data/fakeJob/vocab_embed/orig_cbw_128d_785_embed.npy')
            item_embeds=torch.from_numpy(item_embeds)
            self.embed_layer=nn.Embedding.from_pretrained(item_embeds,freeze=False,
                                                          padding_idx=padding_idx)
        else:
            self.embed_layer=nn.Embedding(input_embed,embed_dim,padding_idx=padding_idx)
        
    def forward(self,item_tensors):
        """
        Compute sequence embed for given item seq.

        Parameters
        ----------
        item_tensors : torch.tensor, optional
        The tensors that you want to get embed.

        Returns
        -------
        batch of embed tensor : torch.FloatTensor
        """
        #item_tensors=[Bs,seqLen]
        embed_tensors=[]

        #using mean pooling to compute item context 
        for tensors in item_tensors:
            #context_tensors=[1,embed_dim] and save them
            context_tensors=self.embed_layer(tensors.long()).sum(dim=0)
            embed_tensors.append(context_tensors)
        
        #concat tensors=[Bs,embed_dim]
        embed_tensors=torch.stack(embed_tensors)

        return embed_tensors

class Attentioner(nn.Module):
    """
    The Attention model for Item embed. Which attention computeing mechansim 
    is Scaled Dot-Product Attention. And the query is items,key & value is 
    description text.
    ...

    Attributes
    ----------
    input_dim : int
        Set the dim of attened tensor.
    hid_dim : int
        Set the dim of transform tensor.

    Methods
    -------
    foward(q_hid,k_hid,v_hid):
        The callable method will perform sacled dot product attention
        on q_hid,k_hid,v_hid.
    """
    def __init__(self,query_dim,key_dim,value_dim,hid_dim):
        """
        Constructs architecture for the attention mechansim.

        Parameters
        ----------
            input_dim : int
                 Set the dim of attened tensor.
            hid_dim : int
                 Set the dim of transform tensor.
        """
        super(Attentioner,self).__init__()
        self.query_dim=query_dim
        self.key_dim=key_dim
        self.value_dim=value_dim
        self.hid_dim=hid_dim

        self.query_layer=nn.Linear(query_dim,hid_dim)
        self.key_layer=nn.Linear(key_dim,hid_dim)
        self.value_layer=nn.Linear(value_dim,hid_dim)
        self.softmax=nn.Softmax(dim=2)
        
    def forward(self,q_hid,k_hid,v_hid):
        """
        The callable method will perform sacled dot product attention
        on q_hid,k_hid,v_hid.And then using weighted sum to get attened
        embed.

        Parameters
        ----------
        q_hid : torch.FloatTensor
            The query tensor that you want atten.
        k_hid : torch.FloatTensor
            The key tensor that attened.
        v_hid : torch.FloatTensor
            The key tensor that will be weighted sum.

        Returns
        -------
        The attened embed for query tensor : torch.FloatTensor
        """
        #q_hid=[Bs,embed_dim]
        #k_hid=[Bs,seqlen,hid_dim]
        #v_hid=[Bs,seqLen,hid_dim]
        q_hid=q_hid.unsqueeze(1)

        #transform 
        q_hid=self.query_layer(q_hid)
        k_hid=self.key_layer(k_hid)
        v_hid=self.value_layer(v_hid)

        #dot-prodcut & scaled then use softmax
        #attned_w=[Bs,1,seqLen]
        attned_w=self.softmax(torch.matmul(q_hid,k_hid.transpose(1,2))/(self.hid_dim**0.5))

        #compute context_vector
        #contexts=[Bs,hid_dim]
        contexts=torch.matmul(attned_w,v_hid).squeeze(1)

        return contexts

            




