import torch
from torch import nn
from torch.nn import functional as F

class fake_classifier(nn.Module):
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
        self.relu=nn.ReLU()
    
    def forward(self,input_tensors):
        hiddens=self.trans_layer(input_tensors)
        for l in self.inter_layer:
            hiddens=self.relu(l(hiddens))
        return self.output_layer(hiddens)

class item_extractor(nn.Module):
    def __init__(self,input_embed,embed_dim,padding_idx,pretrain_weight=None):
        super(item_extractor,self).__init__()
        if pretrain_weight:
            self.embed_layer=nn.Embedding.from_pretrained(pretrain_weight,freeze=False,
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
        #key=[Bs,hid_dim]
        #query=[Bs,hid_dim,seqLen]
        #value=[Bs,seqLen,hid_dim]

        key=key.unsqueeze(1)
        #compute attention weight
        #attn_w=[Bs,1,seqLen]
        attn_w=self.softmax(torch.matmul(key,query))

        #compute context_vector
        #contexts=[Bs,hid_dim]
        contexts=torch.matmul(attn_w,value).squeeze(1)

        return contexts

            




