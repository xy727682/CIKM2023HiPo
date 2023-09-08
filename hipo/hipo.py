import hipo.hipo_fusion_module as hipo_fusion_module
import hipo.mm_fusion_module as mm_fusion_module 
import torch 
import torch.nn as nn

class hipo(nn.Module):                                                                                                
    '''
    The HiPo model settings. 
    '''
    def __init__(self, batch_size, h, dropout, sim_size, bound=[100,100], kernel_size=[5,2], ab='None'):                                                  
        super(hipo, self).__init__()
        '''
        batch_size:     batch size
        h:              Number of heads
        dropout:        dropout rate
        sim_size:       number of historical simialer posts
        bound:          upper bound for determining whether two posts are similar
        kernel_size:    the size of Gaussian kernel
        ab:             ablation mode
        '''
        self.ab = ab
        self.text_fusion = hipo_fusion_module.text_fusion_module(768, 768, 1536, h[0], dropout[0], batch_size, 24, 768, kernel_size)            
        self.image_fusion = hipo_fusion_module.image_fusion_module(1000, 1000, 1000, h[1], dropout[1])                             
        self.env_fusion = hipo_fusion_module.environment_fusion_module(768, 768, 768, h[2], dropout[2], batch = batch_size)                                      
        self.mm_fusion = mm_fusion_module.cnn_edition(ab, h[3], dropout[3])                                                                                                                                                      
        
        self.batch = batch_size
        self.tbound = torch.tensor([bound[0]]).repeat(batch_size,sim_size).cuda()
        self.ibound = torch.tensor([bound[1]]).repeat(batch_size,sim_size).cuda()
        self.dis_pad = torch.ones(self.batch,1).cuda()
        self.sim_size = sim_size 


    def forward(self, word, image, env, env_weight, VSDH, TSDH, tdis=None, idis=None):   
        '''
        input:
            word:   word feature (b_s, n, sentence_length, word_feature)
            image:  spatial feature (b_s, n, spatial_feature)
            env:    environment feature (b_s, n, 3, env_feature)
            env_weight: Historical-Perceptual fusion weight for each historical posts
            VSDH:   VSDH vector of spatial feature
            TSDH:   VSDH vector of textual feature
            tdis:   word feature similarity between two posts
            idis:   spatial feature similarty between two posts
        output:
            result: prediction for optimize
            result2: prediction for visualization
        '''                  
        tdis = torch.le(tdis[:,:self.sim_size],self.tbound).float()
        idis = torch.le(idis[:,:self.sim_size],self.ibound).float()
        dis = torch.cat((self.dis_pad,tdis,idis),dim=-1).view(self.batch,2*self.sim_size+1,1,1)
        word = word*dis
        image = image*dis.view(self.batch,2*self.sim_size+1,1)
        
        text_feat, word_feat, text = self.text_fusion(word)                                   # calling the historical-textual fusion module
        image_feat = self.image_fusion(image)                                                 # calling the historical-spatial fusion module
        self.env_feat, self.dist_feat = self.env_fusion(env, env_weight, VSDH, TSDH, word)    # calling the historical-perceptional fusion module (based on background news)
        result,result2 = self.mm_fusion(word_feat.unsqueeze(1), text_feat.unsqueeze(1), image_feat.unsqueeze(1), self.env_feat.unsqueeze(1), self.dist_feat.unsqueeze(1), self.batch) 
    
        return result,result2
