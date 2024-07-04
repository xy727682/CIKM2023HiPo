import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import pickle
import pandas as pd
from dataset import TrainData_ns, TestData_ns
from torch.utils.data import DataLoader
from hipo_ns.hipo import hipo
from train import train_ns
from rank import eval_ns
import sklearn.metrics as me
from operator import itemgetter
from tqdm import tqdm
import argparse
import random

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        #print(d_model, h , d_k)
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
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class actor(nn.Module):
    def __init__(self, lr, wdc) -> None:
        super(actor, self).__init__()

        self.tatt = ScaledDotProductAttention(768,768,768,1)
        self.iatt = ScaledDotProductAttention(1000,1000,1000,1)
        self.satt = ScaledDotProductAttention(385,385,385,1)
        self.tfc = nn.Linear(768,128)
        self.ifc = nn.Linear(1000,128)
        self.efc = nn.Linear(128,128)
        self.sfc = nn.Linear(385,1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wdc)

    def forward(self, t, i, e, tik):
        tf = self.tfc(self.relu(self.tatt(t[:,1:],t[:,:1],t[:,:1])))
        sf = self.ifc(self.relu(self.iatt(i[:,1:],i[:,:1],i[:,:1])))
        ef = self.efc(e[:,1:])
        af = torch.cat((tf,sf,ef,tik[:,1:]),dim=-1)
        af = self.sig(self.sfc(self.relu(self.satt(af,af,af))))

        return af.squeeze(-1)
            

class critic(nn.Module):
    def __init__(self, lr, wdc):
        super(critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Conv2d(4, 64, 6, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 6, stride=2, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 6, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Flatten(),
            #nn.Linear(256, 512),
            #nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wdc)
    
    def forward(self, tf, sf, hf, tik):
        tik = tik.repeat(1,1,128)
        cf = torch.cat((tf.unsqueeze(1), sf.unsqueeze(1), hf.unsqueeze(1), tik.unsqueeze(1)), dim=1)
        value = self.critic(cf)

        return value.squeeze(-1)

class rl_policy(object):
    def __init__(self, selected_ind, env, rl_lr, wdc) -> None:
        super(rl_policy, self).__init__()

        self.critic = critic(rl_lr[0],wdc[1]).cuda()
        self.actor = actor(rl_lr[1],wdc[2]).cuda()
        self.old_critic = critic(rl_lr[0],wdc[0]).cuda()
        self.old_actor = actor(rl_lr[1],wdc[2]).cuda()
        self.data = []
        self.step = 0
        self.selected_ind = selected_ind
        self.env = env
        self.pool = nn.AdaptiveMaxPool1d(128)
    
    def choose_action(self, s, done, tflag):
        with torch.no_grad():
            if tflag == 'train':
                a = []
                for q in tqdm(range(0,len(self.env.train_text_ind), self.env.batch_size)):
                    self.selected_ind = [n for n in range(q,q+self.env.batch_size)]
                    t,i,e,tic = self.env.data_load(self.selected_ind,s[q:q+self.env.batch_size],done,tflag)
                    action_feat = self.old_actor(t,i,e,tic)
                    try:
                        a.append(torch.multinomial(action_feat, 2, replacement=False))
                    except:
                        a.append(torch.cat((torch.zeros(128,1),torch.ones(128,1)*11),dim=-1))
                    # prob = torch.gather(action_feat, dim=1, index = a)
                    # logp = prob.log()
            if tflag == 'eval':
                a = []
                for q in tqdm(range(0,len(self.env.test_text_ind), self.env.batch_size)):
                    self.selected_ind = [n for n in range(q,q+self.env.batch_size)]
                    t,i,e,tic = self.env.data_load(self.selected_ind,s[q:q+self.env.batch_size],done,tflag)
                    action_feat = self.old_actor(t,i,e,tic)
                    a.append(torch.multinomial(action_feat, 2, replacement=False))
                
            return a
    
    def push_data(self, transitions, i=0):
        if i == 1:
            self.data = []
            return
        self.data.append(transitions)

    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(torch.tensor(s, dtype=torch.float))
            l_a.append(torch.tensor(a[0].clone().detach().cpu(), dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor(s_, dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        self.data = []

        return s, a, r, s_, done
    
    def updata(self):
        self.step += 1
        s, a, ar, s_, adone = self.sample()
        GAMMA = 0.99
        ar = ar.cuda()
        adone = adone.cuda()
        ar = ar.view(s_.shape[0]*s_.shape[1])
        adone = adone.repeat(1,1536).view(s_.shape[0]*s_.shape[1])
        for _ in tqdm(range(5)):
            self.selected_ind = torch.randperm(len(self.env.train_text_ind)*s_.shape[0])[:128]
            ot,oi,oe,otic = self.env.data_load(self.selected_ind.numpy()//s_.shape[1],s,1)
            t,i,e,tic = self.env.data_load(self.selected_ind.numpy()//s_.shape[1],s_.view(s_.shape[0]*s_.shape[1],-1),1)
            tic[:] = ((self.selected_ind // 1536) % 3).view(128,1,1).repeat(1,11,1) + 1
            otic[:] = torch.clone(tic)
            r = ar[self.selected_ind]
            done = adone[self.selected_ind]
            A = []
            #for j in range(1,4):                                                    # j: 第 j 次进行替换
            with torch.no_grad():
                '''loss_v'''
                v1 = self.old_critic(self.pool(t),self.pool(i),e,tic)               # v1: old critic对当前 (text, image, env) 拼接向量的预测
                td_target = r + GAMMA * v1 * (1 - done)                   # td target: 当前状态的 V-value (state value) 
                                                                                # done: 是否是当前epoch的最后一个。如果done=1, 循环终止。
                '''loss_pi'''
                action_feat = self.old_actor(ot,oi,oe,otic)                             # action feat: Feature distribution of the return action. 
                a = torch.multinomial(action_feat, 2, replacement=False)        #              the real action is sampled by multinomial().
                prob = torch.gather(action_feat, dim=1, index = a)
                log_prob_old = prob.log()
                v2 = self.critic(self.pool(ot),self.pool(oi),oe,otic)                   # v2: 目前（完成当前组训练中一部分后）critic (text, img, env)
                v3 = self.critic(self.pool(t),self.pool(i),e,tic)                # v3: 目前critic( old text,img,env) 历史采样的一个，用来？？
                td_error = r + GAMMA * v3 * (1 - done) - v2         # td error: Time Difference error for V-value based estimation
                td_error = td_error.cpu().detach().numpy()                      #           GPU -> CPU, tensor -> numpy
                
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * GAMMA * 0.95 + td                            # TD-based advantage, GAMMA fixed, lambda = 0.95
                    A.append(adv)
                A.reverse()
            A = torch.tensor(A, dtype=torch.float).reshape(-1, 1).cuda()
    
            action_feat = self.actor(ot,oi,oe,otic)
            a = torch.multinomial(action_feat, 2, replacement=False)
            prob = torch.gather(action_feat, dim=1, index = a)
            log_prob_new = prob.log()
            ratio = torch.mean(torch.exp(log_prob_new - log_prob_old))
            L1 = ratio * A                                                      # normal advantage, Loss1
            L2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * A                       # PPO advantage, Loss2
            loss_pi = -torch.min(L1, L2).mean()                                 # loss_policy = - min (Loss1, Loss2), uniform sampling from the batch Loss1&2
            self.actor.optim.zero_grad()
            loss_pi.backward()
            self.actor.optim.step()                                             # step in

            loss_v = F.mse_loss(td_target.detach(), self.critic(self.pool(t),self.pool(i),e,tic))

            self.critic.optim.zero_grad()
            loss_v.backward()
            self.critic.optim.step()
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
        value1 = torch.mean(v1)
        value2 = torch.mean(v2)
        value3 = torch.mean(v3)
        reward = torch.mean(r)
        with open('./result_hipo_rl.txt', 'a') as save_file:
                #save_file.write(str(args))
                save_file.write(' & Old_Critic_New_News: {0:.4f} New_Critic_Old_News: {1:.4f} New_Value_New_News: {2:.4f} Reward: {3:.4f}-----------'.format(value1,value2,value3,reward))
                save_file.write('\r\n')


    def save(self):
        torch.save(self.actor.state_dict(), 'pi.pth')
        torch.save(self.critic.state_dict(), 'v.pth')
    
    def load(self):
        try:
            self.pi.load_state_dict(torch.load('pi.pth'))
            self.v.load_state_dict(torch.load('v.pth'))
            print('...load...')
        except:
            pass


class environment(object):
    def __init__(self,batch_size, args):
        self.train_text_ind = np.load('../hipo_ifnd/hipo/datasets/ifnd/text/text_train_ind_knn.npy').astype(int)
        self.test_text_ind = np.load('../hipo_ifnd/hipo/datasets/ifnd/text/text_valid_ind_knn.npy').astype(int)
        self.train_img_ind = np.load('../hipo_ifnd/hipo/datasets/ifnd/image/image_train_ind_knn.npy').astype(int)
        self.test_img_ind = np.load('../hipo_ifnd/hipo/datasets/ifnd/image/image_valid_ind_knn.npy').astype(int)
    
        self.temp_train_text_ind = self.train_text_ind[:,:6]
        self.temp_test_text_ind = self.test_text_ind[:,:6]
        self.temp_train_img_ind = self.train_img_ind[:,:6]
        self.temp_test_img_ind = self.test_img_ind[:,:6]

        self.train_ind = np.concatenate((self.temp_train_text_ind,self.temp_train_img_ind),axis=-1)
        self.test_ind = np.concatenate((self.temp_test_text_ind,self.temp_test_img_ind),axis=-1)

        self.count = 6

        self.train_img = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/image/trainfeat.npy')).float()
        self.test_img = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/image/valfeat.npy')).float()
        self.env_img = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/image/envfeat.npy')).float()

        self.label_train = torch.tensor([pd.read_csv('../hipo_ifnd/hipo/datasets/ifnd/IFND_train.csv')['label']]).float().permute(1,0)
        self.label_test = torch.tensor([pd.read_csv('../hipo_ifnd/hipo/datasets/ifnd/IFND_valid.csv')['label']]).float().permute(1,0)
        
        self.word_train = pd.read_pickle('../hipo_ifnd/hipo/datasets/ifnd/text/train_word_feat_dict.pkl')
        self.word_valid = pd.read_pickle('../hipo_ifnd/hipo/datasets/ifnd/text/valid_word_feat_dict.pkl')
        self.word_env = pd.read_pickle('../hipo_ifnd/hipo/datasets/ifnd/text/env_word_feat_dict.pkl')

        self.bound_train = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/text/train_bound.npy')).float()
        self.bound_valid = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/text/valid_bound.npy')).float()
        self.TSDH_train = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/text/train_TSHD_128.npy')).float()
        self.TSDH_valid = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/text/valid_TSHD_128.npy')).float()
        self.VSDH_train = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/image/train_VSHD_128.npy')).float()
        self.VSDH_valid = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/image/valid_VSHD_128.npy')).float()
        self.weight_train = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/text/train_weight.npy')).float()
        self.weight_valid = torch.from_numpy(np.load('../hipo_ifnd/hipo/datasets/ifnd/text/valid_weight.npy')).float()
        
        self.batch_size = batch_size
        self.model = hipo(batch_size, args.head_num, args.dropout, 5, transfer = 'twitter').cuda()
        self.optimizer = torch.optim.Adam([{'params':self.model.parameters(), 'lr': args.l_r, 'weight_decay': args.wdc[0]}])
        self.croloss = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]).cuda())
        self.train_epoch = 0
        self.temp = 0
        self.times = 0
        self.temp1= 0
        self.temp2= 0
        self.temp3= 0
        self.temp4= 0
        self.temp5= 0
        self.pool = nn.AdaptiveAvgPool2d((1,768))
        self.args = args

        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)    
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    def reset(self, idx, i):
        self.count = 6
        if idx is None:
            self.train_ind = np.concatenate((self.temp_train_text_ind,self.temp_train_img_ind),axis=-1)
            self.test_ind = np.concatenate((self.temp_test_text_ind,self.temp_test_img_ind),axis=-1)
            
            return np.copy(self.train_ind), np.copy(self.test_ind)
        
        return self.train_ind[idx]
    
    def step(self, action, idx, tflag):
        if tflag == 'choose':
            for i in range(len(action[0])):
                for j in range(len(action[0][0])):
                    tidx = i*len(action[0][0])+j
                    self.train_ind[tidx,action[0][i][j,0]] = torch.clone(torch.from_numpy(self.train_ind[tidx,5:6]))
                    self.train_ind[tidx,5] = torch.clone(torch.from_numpy(self.train_text_ind[tidx,self.count:self.count+1]))
                    self.train_ind[tidx,action[0][i][j,1]] = torch.clone(torch.from_numpy(self.train_ind[tidx,-1:]))
                    self.train_ind[tidx,-1] = torch.clone(torch.from_numpy(self.train_img_ind[tidx,self.count:self.count+1]))
            self.count += 1
            self.count = self.count%100

            return self.train_ind[idx], np.zeros(1536), False
        
        if tflag == 'train':
            self.train_epoch+=1
            if action != []:
                for i in range(len(action[0])):
                    for j in range(len(action[0][0])):
                        tidx = i*len(action[0][0])+j
                        self.train_ind[tidx,action[0][i][j,0]] = torch.clone(torch.from_numpy(self.train_ind[tidx,5:6]))
                        self.train_ind[tidx,5] = torch.clone(torch.from_numpy(self.train_text_ind[tidx,self.count:self.count+1]))
                        self.train_ind[tidx,action[0][i][j,1]] = torch.clone(torch.from_numpy(self.train_ind[tidx,-1:]))
                        self.train_ind[tidx,-1] = torch.clone(torch.from_numpy(self.train_img_ind[tidx,self.count:self.count+1]))
            self.count += 1
            self.count = self.count%100
            train_dataset = TrainData_ns(self.train_ind[:,:5], self.train_img, self.train_ind[:,6:-1], self.env_img, self.label_train, self.word_train, self.word_env, self.bound_train, self.TSDH_train, self.VSDH_train, self.weight_train, 5)
            train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=5, drop_last=True)    
            loss, res, label, index = train_ns(self.train_epoch, len(train_dataset), train_dataloader, self.model, self.optimizer, self.croloss, self.batch_size, 'hipo')
            index = np.array(index).flatten()
            label = np.array(label).flatten()
            res = np.array(res)[:,:,1].flatten()
            predi = np.round(res)
            predi = np.nan_to_num(predi,nan=0.5)
            acc = me.accuracy_score(label,predi)
            recall = me.recall_score(label,predi)
            preci = me.precision_score(label,predi)
            auc = me.roc_auc_score(label,res)
            f1 = me.f1_score(label,predi)
            print('acc: ', acc,'recall: ',recall,'prec: ',preci, 'auc: ',auc,'f1: ',f1)
            r = np.ones(1536)*0.5
            for i in range(len(index)):
                if predi[i] == label[i]:
                    r[index[i]] = 1
                else:
                    r[index[i]] = 0

        
        if tflag == 'eval':
            test_epoch = self.train_epoch
            if action != []:
                for i in range(len(action[0])):
                    for j in range(len(action[0][0])):
                        tidx = i*len(action[0][0])+j
                        self.test_ind[tidx,action[0][i][j,0]] = torch.clone(torch.from_numpy(self.test_ind[tidx,5:6]))
                        self.test_ind[tidx,5] = torch.clone(torch.from_numpy(self.test_text_ind[tidx,self.count:self.count+1]))
                        self.test_ind[tidx,action[0][i][j,1]] = torch.clone(torch.from_numpy(self.test_ind[tidx,-1:]))
                        self.test_ind[tidx,-1] = torch.clone(torch.from_numpy(self.test_img_ind[tidx,self.count:self.count+1]))
            self.count += 1
            self.count = self.count%100
            test_dataset = TestData_ns(self.test_ind[:,:5], self.test_img, self.test_ind[:,6:-1], self.env_img, self.label_test, self.word_valid, self.word_env, self.bound_valid, self.TSDH_valid, self.VSDH_valid, self.weight_valid, 5)
            test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle=True, num_workers=5, drop_last=True)    
            eva_loss, acc, recall, preci, auc, f1 = eval_ns(len(test_dataset), test_dataloader, self.model, self.optimizer, self.croloss, self.batch_size, 1, 'hipo')
            r = eva_loss
            print('eva_loss: ', eva_loss, 'eva_acc: ',acc, '12')
            if acc > self.temp:
                self.temp = acc
                self.times = test_epoch
                self.temp2 = recall
                self.temp3 = preci
                self.temp4 = auc
                self.temp5 = f1
            if test_epoch % 10 == 0:
                with open('./result_hipo_rl.txt', 'a') as save_file:
                    #save_file.write(str(args))
                    save_file.write('-----------Test Loss: {0:.4f} Fin_Acc: {1:.4f} Max_Acc: {2:.4f} Max_Epoch: {3:.1f} recall: {4:.4f} precision: {5:.4f} AUC: {6:.4f} f1: {7:.4f}-----------'.format(eva_loss, acc, self.temp, self.times, self.temp2, self.temp3, self.temp4, self.temp5))
                    save_file.write('\r\n')
                    print(self.times, self.temp)
           
            
        #r = 1
        
        # if self.count == 16 or self.count == self.train_text_ind.shape[-1]:
        #     done = True
        # else:
        #     done = False

        return self.train_ind[idx], r, True
    
    def data_load(self, idx1, idx2, done, tflag = 'default'):
        if tflag == 'train':
            idx2 = idx2.long()
            text = np.zeros((len(idx1),11,24,768))
            image = torch.zeros((len(idx1),11,1000))
            env = torch.zeros((len(idx1),11,128))
                              
            for i in range(len(idx1)):
                text[i,0] = self.word_train[str(idx1[i])]
                image[i,0] = self.train_img[idx1[i]]
                env[i] = self.VSDH_train[idx1[i]].repeat(1,11,1)[:,:,:128]
                wkeys = self.train_ind[idx1[i]][:5].astype(str)
                text[i,1:6] = itemgetter(*wkeys)(self.word_env)
                ikeys = self.train_ind[idx1[i]][6:-1].astype(str)
                text[i,6:] = itemgetter(*ikeys)(self.word_env)
                image[i,1:6] = self.env_img[idx2[i,:5]]
                image[i,6:] = self.env_img[idx2[i,6:-1]]
            text = torch.mean(torch.from_numpy(text),dim=-2)

        elif tflag == 'eval':
            idx2 = idx2.long()
            text = np.zeros((len(idx1),11,24,768))
            image = torch.zeros((len(idx1),11,1000))
            env = torch.zeros((len(idx1),11,128))
            
            for i in range(len(idx1)):
                text[i,0] = self.word_valid[str(idx1[i])]
                image[i,0] = self.test_img[idx1[i]]
                env[i] = self.VSDH_valid[idx1[i]].repeat(1,11,1)[:,:,:128]
                wkeys = self.test_ind[idx1[i]][:5].astype(str)
                text[i,1:6] = itemgetter(*wkeys)(self.word_env)
                ikeys = self.test_ind[idx1[i]][6:-1].astype(str)
                text[i,6:] = itemgetter(*ikeys)(self.word_env)
                image[i,1:6] = self.env_img[idx2[i,:5]]
                image[i,6:] = self.env_img[idx2[i,6:-1]]
            text = torch.mean(torch.from_numpy(text),dim=-2)
        
        else:
            idx2 = idx2.long()
            text = np.zeros((len(idx1),11,24,768))
            image = torch.zeros((len(idx1),11,1000))
            env = torch.zeros((len(idx1),11,128))
                              
            for i in range(len(idx1)):
                text[i,0] = self.word_train[str(idx1[i])]
                image[i,0] = self.train_img[idx1[i]]
                env[i] = self.VSDH_train[idx1[i]].repeat(1,11,1)[:,:,:128]
                wkeys = self.train_ind[idx1[i]][:5].astype(str)
                text[i,1:6] = itemgetter(*wkeys)(self.word_env)
                ikeys = self.train_ind[idx1[i]][6:-1].astype(str)
                text[i,6:] = itemgetter(*ikeys)(self.word_env)
                image[i,1:6] = self.env_img[idx2[idx1[i],:5]]
                image[i,6:] = self.env_img[idx2[idx1[i],6:-1]]
            text = torch.mean(torch.from_numpy(text),dim=-2)

        return text.cuda().float(), image.cuda().float(), env.cuda().float(), (torch.ones((128,11,1))*done).cuda()

def main(env, agent):
    idx = None
    #agent = rl_policy(idx,env,env.args.rl_lr,env.args.wdc)
    #agent.load()
    max_rewards = -1000000
    for _ in range(50):
        s,se = env.reset(idx,_)
        env.step([],idx,tflag='train')
        env.step([],idx,tflag='eval')
    for _ in range(30):
        #s,se = env.reset(idx,_)
        #start = True
        rewards = 0
        agent.push_data([],1)
        for ii in range(5):
        #while start:
            s,se = env.reset(idx,_)
            for i in range(3):
                #env.render()
                if i != 2:
                    a = agent.choose_action(torch.tensor(s, dtype=torch.float),i+1,tflag='train') 
                    s_t, r, done = env.step([a],idx,tflag='choose')
                    agent.push_data((s, a, r , s_t, done))
                    s = np.copy(s_t.squeeze(0))
                else:
                    a = agent.choose_action(torch.tensor(s_t.squeeze(0), dtype=torch.float),i+1,tflag='train')
                    s_t, r, done = env.step([a],idx,tflag='train')
                    agent.push_data((s, a, r , s_t, done))
                    s = np.copy(s_t.squeeze(0))
                    a = agent.choose_action(torch.tensor(se, dtype=torch.float),i-1,tflag='eval')
                    se, re, de = env.step([a],idx,tflag='choose')
                    a = agent.choose_action(torch.tensor(s_t.squeeze(0), dtype=torch.float),i,tflag='eval')
                    se, re, de = env.step([a],idx,tflag='choose')
                    a = agent.choose_action(torch.tensor(s_t.squeeze(0), dtype=torch.float),i+1,tflag='eval')   
                    se, re, de = env.step([a],idx,tflag='eval')
                rewards += r
                # agent.push_data((s, a, r , s_t, done),i)
                # s = np.copy(s_t.squeeze(0))
                # if done:
                #     #start = False
                #     break
                # #if i % 1 == 0:
                # a = agent.choose_action(torch.tensor(se, dtype=torch.float),tflag='eval')
                # s_ = env.step([a],idx,tflag='choose')
                # a = agent.choose_action(torch.tensor(s_.squeeze(0), dtype=torch.float),tflag='eval')
                # s_ = env.step([a],idx,tflag='choose') 
                # a = agent.choose_action(torch.tensor(s_.squeeze(0), dtype=torch.float),tflag='eval')   
                # s_, r, done = env.step([a],idx,tflag='eval')
                # #s = s_t.squeeze(0)
                # se = s_.squeeze(0)
        agent.updata()
            # if i == 9:
            #     start = False
        # if _ % 10 == 0:
        #rewards += r
        rewards = np.mean(rewards)
        print(_, ' ', rewards, ' ', agent.step)
        if max_rewards < rewards:
            max_rewards = rewards
            agent.save()
    with open('./result_hipo_rl.txt', 'a') as save_file:
                    #save_file.write(str(args))
                    save_file.write('\r\n')

def init_args(i,j):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    
    parser.add_argument('--l_r', type=float, default=5e-6, help='Learning rate.')
    parser.add_argument('--rl_lr', type=float, default=[2e-6,1e-5], help='Critic learning rate.')
    parser.add_argument('--dropout', type=float, default=[0,0.1,0], help='text/image/env dropout.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--sim_size', type = int, default= 5, help='sim_size')
    parser.add_argument('--head_num', type = int, default= [8,2,4], help='Number of text/image/env heads')
    parser.add_argument('--bound', type = float, default= [100,100], help='text/image knn bound')
    parser.add_argument('--wdc', type = float, default= [1e-2,1e-4,1e-4], help='weight decay')
    parser.add_argument('--kernel_size', type = int, default= [5,2], help='kernel size of word kernel attention')
    
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=5, help='Workers number.')
    parser.add_argument('--dataset', type=str, default='ifnd', help='Chose the dataset')
    parser.add_argument('--model_name', type=str, default='hipo', help='model name')
    parser.add_argument('--model', type = str, default='multi',help='model')

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    # a = torch.rand((64,1,768))
    # b = torch.rand((64,1,1000))
    # c = torch.rand((64,1,128))
    # d = torch.rand((64,12,768))
    # e = torch.rand((64,12,1000))
    # ind = torch.randperm(12)
    # ind = ind.unsqueeze(0).repeat(64,1)
    for i in (1e-4,1e-5,1e-6,0):
        for j in (1e-4,1e-5,1e-6,0):
            #if i == 1e-3 and j == 1e-3:
            #    continue
            # i = 0
            # j = 0
            args = init_args(i,j)
            seed = args.seed
            random.seed(seed)
            np.random.seed(seed)    
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            env = environment(128,args)
            agent = rl_policy(None,env,env.args.rl_lr,env.args.wdc)
            main(env, agent)