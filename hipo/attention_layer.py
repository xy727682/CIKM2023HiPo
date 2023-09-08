import numpy as np
import torch
from torch import nn
from torch.nn import init


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        d_model:    the dimension of the model output
        d_k:        the dimension of queries and keys
        d_v:        the dimension of values
        h:          attention head count
        dropout:    dropout rate
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k) 
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_v, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout) 

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        input:
            queries:            Queries (b_s, nq, d_model)
            keys:               Keys (b_s, nk, d_model)
            values:             Values (b_s, nk, d_model)
            attention_mask:     Mask over attention values (b_s, h, nq, nk). True for being masked.
            attention_weights:  Multiplicative weights for attention values (b_s, h, nq, nk).
        output:
            out: Scaled dot-product attention output (b_s, nq, d_model).
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
    

class Depth_Pointwise_Conv1d(nn.Module):
    '''
    Depth Pointwise Conv1d
    '''
    def __init__(self,in_ch,out_ch,k):
        '''
        in_ch:  In channels count
        out_ch: Out channels count
        k:      kernel size
        '''
        super().__init__()
        if(k==1):
            self.depth_conv=nn.Identity()
        else:
            self.depth_conv=nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k//2
                )
        self.pointwise_conv=nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )
    def forward(self,x):
        '''
        input:
            x:      Feature input from previous layer
        output:
            out:    Depth Pointwise Conv1d result
        '''
        out=self.pointwise_conv(self.depth_conv(x))
        return out
    

class MUSEAttention(nn.Module):
    '''
    MUSE attention
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        d_model:    the dimension of the model output
        d_k:        the dimension of queries and keys
        d_v:        the dimension of values
        h:          Number of heads
        dropout:    dropout rate
        '''
        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.conv1=Depth_Pointwise_Conv1d(h * d_v, d_model,1)
        self.conv3=Depth_Pointwise_Conv1d(h * d_v, d_model,3)
        self.conv5=Depth_Pointwise_Conv1d(h * d_v, d_model,5)
        self.dy_paras=nn.Parameter(torch.ones(3))
        self.softmax=nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        input:
            queries:            Queries (b_s, nq, d_model)
            keys:               Keys (b_s, nk, d_model)
            values:             Values (b_s, nk, d_model)
            attention_mask:     Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights:  Multiplicative weights for attention values (b_s, h, nq, nk).
        output:
            out: Scaled dot-product attention output (b_s, nq, d_model).
        '''
        #Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2=v.permute(0,1,3,2).contiguous().view(b_s,-1,nk) # (bs, dim, n)
        self.dy_paras=nn.Parameter(self.softmax(self.dy_paras))
        out2=self.dy_paras[0]*self.conv1(v2)+self.dy_paras[1]*self.conv3(v2)+self.dy_paras[2]*self.conv5(v2)
        out2=out2.permute(0,2,1) # (bs, n, dim)

        out=out+out2
        return out


class text_attention_layer(nn.Module):
    '''
    text attention layer for historical text fusion module
    '''
    def __init__(self, d_model, d_k, d_v, h=8, dropout=.1):
        '''
        d_model:    the dimension of the model output
        d_k:        the dimension of queries and keys
        d_v:        the dimension of values
        h:          Number of heads
        dropout:    dropout rate
        '''
        super(text_attention_layer, self).__init__()

        self.attention = ScaledDotProductAttention(d_model, d_k, d_v, h, dropout) 
        self.relu = nn.ReLU()                                                     
    
    def forward(self, text):
        '''
        input:
            text:   Text feature output from the previous kernel attention layer (b_s, p, semantic_feat).
        output:
            output: (b_s, semantic_feat)
        '''
        text = text.squeeze(2)                                      
        value = torch.cat((text[:,0:1,:], text[:,0:1,:]), dim=-1)   
        key = text[:,0:1,:]                                         
        query = text[:,1:2,:]                                       
        output = self.attention(query, key, value)                  

        for i in range(2,len(text[0])):                            
            value = torch.cat((key, output), dim=-1)                
            query = text[:,i:i+1,:]                                 
            output = self.attention(query, key, value)             
        output = self.relu(output)                                  

        return output.squeeze(1)                                    


class image_attention_layer(nn.Module):
    '''
    the image attention layer for historical image and environment fusion module
    '''
    def __init__(self, d_model, d_k, d_v, h=8, dropout=.1):
        '''
        d_model:    the dimension of the model
        d_k:        the dimension of queries and keys
        d_v:        the dimension of values
        h:          Number of heads
        dropout:    dropout rate
        '''
        super(image_attention_layer, self).__init__()

        self.attention = ScaledDotProductAttention(d_model, d_k, d_v, h, dropout) 
        self.relu = nn.ReLU()                                                     

    def forward(self, image):
        '''
        input:
            image: Image feature output from vgg19 (b_s, p, image_feat) or environment features (b_s, 5, environment_feat).
        output:
            output: (b_s, image_feat) or (b_s, environment_feat)
        '''
        value = image[:,0:1,:]                         
        key = image[:,0:1,:]                           
        query = image[:,1:,:]                          
        output = self.attention(query, key, value)     
        output = self.relu(output)                     

        return output

if __name__=='__main__':
    # test
    model = text_attention_layer(768, 768, 768, 8, 0.1).cuda()
    for i in range(0,1000):
        feat = torch.rand((256,11,768)).cuda()
    print(model(feat))