import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

model_uniConfig={"embed_dim":300,'hid_dim':512,"fc_dim":512,"meta_dim":12,
              "output_dim":1,"fc_layerN":2,
              "dropout_rate":0.1,"using_pretrain_weight":True,
             }

class fake_classifier(nn.Module):
    """The final sub model for output fake prob.

    This model input is hidden state of other model extracted.
    and then we  output 1d logitics vec using N dense layer and classifier.
    If you want to get fake prob,you can use sigmoid function to produce prob.



    """

    def __init__(self,concat_hid,inter_hid,output_hid,n_layers):
        super(fake_classifier,self).__init__()
        #build module dim attr
        self.concat_hid=concat_hid
        self.inter_hid=inter_hid
        self.output_hid=output_hid

        self.trans_layer=nn.Linear(concat_hid,inter_hid)
        #build intermediate fc layer
        self.inter_layer=nn.ModuleList([nn.Linear(inter_hid//pow(2,i),inter_hid//pow(2,i+1)) 
                                        for i in range(n_layers)])
        #build output layer
        self.output_layer=nn.Linear(inter_hid//pow(2,n_layers),output_hid)

        #build activation func
        self.leakyRelu=nn.LeakyReLU()
    
    def forward(self,input_tensors):
        hiddens=self.leakyRelu(self.trans_layer(input_tensors))
        for l in self.inter_layer:
            hiddens=self.leakyRelu(l(hiddens))
        return self.output_layer(hiddens)

class item_extractor(nn.Module):
    def __init__(self,input_embed,embed_dim,padding_idx,using_pretrain_weight=False):
        super(item_extractor,self).__init__()
        if using_pretrain_weight:
            item_embeds=np.load(r'./Data/fakeJob/vocab_embed/fastText_300d_1061_embed.npy')
            item_embeds=torch.from_numpy(item_embeds)
            self.embed_layer=nn.Embedding.from_pretrained(item_embeds,freeze=False,
                                                          padding_idx=padding_idx)
        else:
            self.embed_layer=nn.Embedding(input_embed,embed_dim,padding_idx=padding_idx)
        
    def forward(self,item_tensors):
        #item_tensors=[Bs,seqLen]
        embed_tensors=[]

        #using mean pooling to compute item context 
        for tensors in item_tensors:
            #context_tensors=[1,embed_dim] and save them
            context_tensors=self.embed_layer(tensors).mean(dim=0)
            embed_tensors.append(context_tensors)
        
        #concat tensors=[Bs,embed_dim]
        embed_tensors=torch.stack(embed_tensors)

        return embed_tensors

class Attentioner(nn.Module):
    def __init__(self):
        super(Attentioner,self).__init__()
        self.softmax=nn.Softmax(dim=2)
        
    def forward(self,key,query,value):
        #key=[Bs,embed_dim]
        #query=[Bs,seqLen,hid_dim]
        #value=[Bs,seqLen,hid_dim]
        key=key.unsqueeze(1)
        #compute attention weight
        #attn_w=[Bs,1,seqLen]
        attn_w=self.softmax(torch.matmul(key,query))

        #compute context_vector
        #contexts=[Bs,hid_dim]
        contexts=torch.matmul(attn_w,value).squeeze(1)

        return contexts

            




