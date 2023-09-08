import numpy as np
import torch
from torch import nn
from torch.nn import init


class kernel_layer(nn.Module):
    '''
    kernel attention layer
    '''                                                                   
    def __init__(self, batch_size, sentence_length, kernel_size) -> None:                       
        super(kernel_layer,self).__init__()
        '''
        batch_size: batch size
        sentence_length: the number of word in each sentence
        kernel_size: the size of Gaussian kernel
        '''
        self.cos = nn.CosineSimilarity(dim=-1)                                                
        self.kernel = self.get_kernel(kernel_size[0],2,1)                                        
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)                  
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)                       
        self.BN=nn.BatchNorm2d(num_features=1)                                                  
        self.translation = torch.zeros((batch_size, sentence_length, sentence_length)).cuda()  
        self.kernel_linear = nn.Linear(sentence_length,sentence_length)                        
        self.kernel_soft = nn.Softmax(dim=-2)                                                  
        self.pad = kernel_size[1]

    def forward(self, news, post):  
        '''
        input:
            news: the feature of the post to be detected (b_s, sentence_length, word_feat)
            post: the feature of the historical posts (b_s, sentence_length, word_feat)
        output:
            word_feat: the feature of the post to be detected processed after kernel attention (b_s, sentence_length, word_feat)
            post_feat: the feature of the historical posts processed after kernel attention (b_s, sentence_length, word_feat)
        '''                                                     
        for i in range(0,len(news[0])):
            temp = self.cos(news, post).unsqueeze(1)
            self.translation[:,i:i+1] = temp
        self.translation = self.translation.unsqueeze(1)
        word_kernel = nn.functional.conv2d(self.translation, self.weight, padding = self.pad)
        word_kernel = self.BN(word_kernel).squeeze(1)
        word_kernel = self.kernel_linear(word_kernel)
        word_kernel = self.kernel_soft(word_kernel).sum(dim=-1).unsqueeze(-1)
        word_feat = word_kernel*news
        word_feat = word_feat.mean(dim=1)
        post_feat = word_kernel*post
        post_feat = post_feat.mean(dim=1).unsqueeze(1)
        self.translation = self.translation.squeeze(1)                                    

        return word_feat, post_feat

    def get_kernel(self, kernel_size, sigma, k):
        '''
        calculate Gaussian kernel
        '''                                         
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        X = np.linspace(-k, k, kernel_size)
        Y = np.linspace(-k, k, kernel_size)
        x, y = np.meshgrid(X, Y)
        x0 = 0
        y0 = 0
        gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
        
        return gauss


class text_kernel_attention(nn.Module):       
    '''
    text kernel attention module
    '''                                   
    def __init__(self, batch_size, sentence_length, kernel_size, wdrop=0.5) -> None:      
        super(text_kernel_attention,self).__init__()
        '''
        batch_size: batch size
        sentence_length: the number of word in each sentence
        kernel_size: the size of Gaussian kernel
        wdrop: dropout rate
        '''
        self.kernel1 = kernel_layer(batch_size, sentence_length, kernel_size)   
        self.drop1 = nn.Dropout(wdrop)
        self.norm1 = nn.BatchNorm1d(768)
        self.klinear1 = nn.Linear(768,768)                                 
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        
    def forward(self, words):                       
        '''
        input:
            words: words feature provided by BERT (b_s, n, sentence_length, word_feat)
        output:
            word_feat: the fusion semantic feature of the post to be detected processed after kernel attention layer (b_s, 1, word_feat)
            post_feat: the semantic feature of the historical posts processed after kernel attention layer (b_s, n-1, word_feat)
        '''
        news = words[:,0]
        post = words[:,1]

        # Calling the kernel attention layer
        word_feat,post_feat = self.kernel1(news,post)

        # use the same layer in a loop
        for i in range(2,len(words[0])):
            post = words[:,i]
            temp_feat,p = self.kernel1(news,post)
            word_feat = word_feat + temp_feat
            post_feat = torch.cat((post_feat,p),dim=1)
        #word_feat = word_feat/len(words[0])
        post_feat = torch.cat((word_feat.unsqueeze(1),post_feat),dim=1)
        word_feat = self.relu(self.drop1(self.klinear1(word_feat/len(words[0]))))
        
        return word_feat,post_feat

if __name__=='__main__':
    from tqdm import tqdm
    # test
    model = text_kernel_attention(256, 24, [5,2], 0.5).cuda()
    feat = torch.rand((256,11,24,768)).cuda()
    for i in tqdm(range(0,100)):
        res = model(feat)
    print(res)