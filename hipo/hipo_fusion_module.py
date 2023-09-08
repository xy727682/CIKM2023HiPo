import attention_layer as attention_layer
import kernel_attention_layer as kernel_attention_layer
import torch
import torch.nn as nn
from tqdm import tqdm


class text_fusion_module(nn.Module):                                                                                       
    '''
    Historical-Textual fusion module.
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout, batch_size, sentence_length, word_length, kernel_size=[5,2]):                           
        super(text_fusion_module,self).__init__()
        '''
        d_model: Output dimensionality of the model
        d_k: Dimensionality of queries and keys
        d_v: Dimensionality of values
        h: Number of heads
        dropout: dropout rate
        batch_size: batch size
        sentence_length: the number of word in each sentence
        word_length: the length of word feature
        kernel_size: the size of Gaussian kernel
        '''
        self.text_attention = attention_layer.text_attention_layer(d_model, d_k, d_v, h, dropout)                         
        self.kernel_attention = kernel_attention_layer.text_kernel_attention(batch_size, sentence_length, kernel_size)   
        self.semantic_linear = nn.Linear(d_model, 768)                                                                    
        self.semantic_linear1 = nn.Linear(d_model, 128) 
        self.word_linear = nn.Linear(word_length, 768)                                                                 
        self.word_linear1 = nn.Linear(word_length, 128)

    def forward(self, word):                                    
        '''
        input:
            word: Word feature of the news to be detected and it's similar historical posts (b_s, n, sentence_length, 768).
        output:
            semantic_feat: text level historical fusion word feature processed by text attention layer
            word_feat: word level historical fusion word feature processed by kernel attention layer
            temp: word feature of historical posts processed by kernel attention layer
        '''
        word_feat,temp = self.kernel_attention(word)
        semantic_feat = self.text_attention(temp)
        semantic_feat = torch.tanh(self.semantic_linear1(semantic_feat))   
        word_feat = torch.tanh(self.word_linear1(word_feat))               

        return semantic_feat, word_feat, temp


class image_fusion_module(nn.Module):                                                            
    '''
    Historical-Spatial fusion module.
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout):                                         
        super(image_fusion_module,self).__init__()
        '''
        d_model: Output dimensionality of the model
        d_k: Dimensionality of queries and keys
        d_v: Dimensionality of values
        h: Number of heads
        dropout: dropout rate
        '''
        self.image_attention = attention_layer.image_attention_layer(d_model, d_k, d_v, h, dropout)   
        self.avgpool = nn.AdaptiveAvgPool2d((1,1000))                                               
        self.image_linear1 = nn.Linear(1000,128)                                                    
        self.image_linear = nn.Linear(1000,1000)                                                  
    
    def forward(self, image):                                     
        '''
        input:
            image: spatial feature of the news to be detected and it's similar historical posts (b_s, n, 1000).
        output:
            image_fusion_feat: text level historical fusion spatial feature processed by image attention layer
        '''
        image_feat = self.image_attention(image)                     
        image_feat = self.avgpool(image_feat).squeeze(1)                 
        image_fusion_feat = torch.tanh(self.image_linear1(image_feat))   

        return image_fusion_feat


class environment_fusion_module(nn.Module):
    '''
    Historical-Perceptual fusion module.
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout, batch = 1):
        super(environment_fusion_module, self).__init__()
        '''
        d_model: Output dimensionality of the model
        d_k: Dimensionality of queries and keys
        d_v: Dimensionality of values
        h: Number of heads
        dropout: dropout rate
        '''

        # Initialize the self-attention layer on background
        self.env_attention = attention_layer.image_attention_layer(d_model, d_k, d_v, h = h, dropout=dropout)

        # Initialize the adaptive average pooling layer, compressing the last 2 dimensions to (1, 768)
        self.avgpool = nn.AdaptiveAvgPool2d((1,768))

        # Initialize the perceptional (based on the news environment) layer
        self.env_linear = nn.Linear(768,128)

        # Initialize the VSDH layer
        self.VSDH_linear = nn.Linear(128,128)

        # Initialize the TSDH layer
        self.TSDH_linear = nn.Linear(128,128)

        # Initialize the ReLU layer
        self.relu = nn.ReLU()

        # Initialize the classifier
        self.softmax = nn.Softmax(dim=-1)

        # Initialize the border linear layer
        self.bound_linear1 = nn.Linear(384*6,768)

        # Initializet the model batch
        self.batch = batch

        self.sdhlinear = nn.Linear(256,128)

    def forward(self, env, env_weight, VSDH, TSDH, text):
        '''
        input:
            env: feature of historical posts processed by kernel attention layer (not used)
            env_weight: Historical-Perceptual fusion weight for each historical posts (not used)
            VSDH: VSDH vector of spatial feature (batch, 128)
            TSDH: VSDH vector of textual feature (batch, 128)
            text: text level historical fusion word feature processed by text attention layer (not used)
        output: 
            env_feat: historical-perceptual fusion feature
            env_dist_feat: historical posts distribution feature
        '''

        # The VSDH feature
        env_dist_feat = self.VSDH_linear(VSDH[:,:128]).squeeze(1)

        # The TSDH feature
        env_feat = self.TSDH_linear(TSDH[:,:128]).squeeze(1)
        
        env_dist_feat = torch.tanh(self.sdhlinear(torch.cat((env_dist_feat,env_feat),dim=-1)))
        env_feat = torch.cat((self.avgpool(text[:,0:1]).squeeze(2), env.squeeze(2)),dim=1)
        env_feat = self.env_attention(env_feat)*self.softmax(env_weight).repeat(1,1,768)
        env_feat = self.bound_linear1(env_feat.view(self.batch,1,384*6)).squeeze(1)
        env_feat = torch.tanh(self.env_linear(env_feat))
        
        return env_feat, env_dist_feat

if __name__=='__main__':
    # test
    model = image_fusion_module(1000,1000,1000, 8, 0.1).cuda()
    feat = torch.rand((256,11,1000)).cuda()
    for i in tqdm(range(0,100)):
        res = model(feat)
    print(res)