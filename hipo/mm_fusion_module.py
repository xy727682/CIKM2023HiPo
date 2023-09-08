import torch
import torch.nn as nn
from hipo.attention_layer import MUSEAttention


class cnn_edition(nn.Module):
    '''
    The multi-modality feature fusion module. 
    '''
    def __init__(self,ab='MUSE',h=8,d=0.1):
        super(cnn_edition,self).__init__()
        '''
        ab: ablation mode
        h: head number
        d: dropout rate
        '''
        if ab == 'MUSE':
            self.attn = MUSEAttention(5,5,5,h,d)
            self.pool = nn.AdaptiveAvgPool2d((1,128))       
        elif ab in ['MUSE2i','MUSE2t','MUSE2e']:
            self.attn = MUSEAttention(2,2,2,h,d)
            self.pool = nn.AdaptiveAvgPool2d((1,128))
        elif ab in ['MUSE2it','MUSE2ie']:
            self.attn = MUSEAttention(3,3,3,h,d)
            self.pool = nn.AdaptiveAvgPool2d((1,128))
        elif ab == 'MUSE2te':
            self.attn = MUSEAttention(4,4,4,h,d)
            self.pool = nn.AdaptiveAvgPool2d((1,128))
        
        # Compute the prediction by applying two linear layers. 
        self.linear1 = nn.Linear(128,10)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(10,2)

        # Use Sigmoid to embed the output to 0 ~ 1. 
        self.sig = nn.Softmax(dim=-1)
        self.ab = ab


    def forward(self, word_feat, semantic_feat, image_feat, env_feat, dist_feat):
        '''
        input:
            *_feat: different features from privious module by size (batch, 1, 128)
        output:
            mm_feat: prediction for optimization
            mm_feat2: prediction for visualization
        '''
        if self.ab == 'MUSE':
            mm_feat = torch.cat((word_feat, semantic_feat, image_feat, env_feat, dist_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)        
        elif self.ab == 'MUSE2i':
            mm_feat = torch.cat((image_feat, image_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)
        elif self.ab == 'MUSE2t':
            mm_feat = torch.cat((word_feat, semantic_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)
        elif self.ab == 'MUSE2e':
            mm_feat = torch.cat((env_feat, dist_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)
        elif self.ab == 'MUSE2it':
            mm_feat = torch.cat((word_feat, semantic_feat, image_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)
        elif self.ab == 'MUSE2ie':
            mm_feat = torch.cat((image_feat, env_feat, dist_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)
        elif self.ab == 'MUSE2te':
            mm_feat = torch.cat((word_feat, semantic_feat, env_feat, dist_feat),dim=1)
            mm_feat = self.pool(self.attn(mm_feat.permute(0,2,1), mm_feat.permute(0,2,1), mm_feat.permute(0,2,1)).permute(0,2,1)).squeeze(1)
        mm_feat = torch.tanh(self.dropout(self.linear1(mm_feat)))
        mm_feat = self.linear2(mm_feat)
        mm_feat2 = self.sig(mm_feat)

        return mm_feat, mm_feat2
