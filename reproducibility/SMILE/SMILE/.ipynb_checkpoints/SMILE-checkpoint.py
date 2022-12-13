# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:17:45 2020

@author: Yang Xu
"""
import gc
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from os.path import join
from .contrastive_loss_pytorch import ContrastiveLoss

##-----------------------------------------------------------------------------
class SMILE(torch.nn.Module):
    def __init__(self,input_dim=2000,clf_out=10):
        super(SMILE, self).__init__()
        self.input_dim = input_dim
        self.clf_out = clf_out
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(128, self.clf_out),
            torch.nn.Softmax(dim=1))
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(128, 32))
        
    def forward(self, x):
        out = self.encoder(x)
        f = self.feature(out)
        y= self.clf(out)
        return f,y

def SMILE_trainer(X, model, 
                  lr=1e-4,
                  batch_size = 512, 
                  num_epoch=5,
                  f_temp = 0.05, 
                  p_temp = 0.15,
                  plot_loss=True,
                  log_dir=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = f_temp)
    p_con = ContrastiveLoss(batch_size = model.clf_out,temperature = p_temp)
    opt = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9,weight_decay=5e-4)
    
    loss_curve = []
    for k in range(num_epoch):
        model.to(device)
        n = X.shape[0]
        r = np.random.permutation(n)
        X_train = X[r,:]
        X_tensor=torch.tensor(X_train).float()
        
        losses = 0
        for j in range(n//batch_size):
            inputs = X_tensor[j*batch_size:(j+1)*batch_size,:].to(device)
            noise_inputs = inputs + torch.normal(0,1,inputs.shape).to(device)
            noise_inputs2 = inputs + torch.normal(0,1,inputs.shape).to(device)
            
            feas,o = model(noise_inputs)
            nfeas,no = model(noise_inputs2)
            
            fea_mi = f_con(feas,nfeas)
            p_mi = p_con(o.T,no.T)
            
            loss = fea_mi + p_mi
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses += loss.data.tolist()

        loss_curve.append(losses)
        print("Total loss: "+str(round(losses,4)))
        gc.collect()

        if log_dir is not None:
            if (k+1) % 10 == 0:  # saving ckpts at interval 10
                torch.save(
                    {
                        'epoch': k + 1,
                        # 'arch': args.arch,
                        'state_dict': model.state_dict(),
                        # 'optimizer' : optimizer.state_dict(),
                    }, 
                    join(log_dir, 'checkpoint_{:04d}.pth.tar'.format(k+1))
                )
    if plot_loss:
#         plt.plot(loss_curve)
#         plt.show()
        return loss_curve
        
    
class Paired_SMILE(torch.nn.Module):
    def __init__(self,input_dim_a=2000,input_dim_b=2000,clf_out=10):
        super(Paired_SMILE, self).__init__()
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.clf_out = clf_out
        self.encoder_a = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_a, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.encoder_b = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_b, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(128, self.clf_out),
            torch.nn.Softmax(dim=1))
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(128, 32))
        
    def forward(self, x_a,x_b):
        out_a = self.encoder_a(x_a)
        f_a = self.feature(out_a)
        y_a = self.clf(out_a)
        
        out_b = self.encoder_b(x_b)
        f_b = self.feature(out_b)
        y_b = self.clf(out_b)
        return f_a,y_a,f_b,y_b
    
def PairedSMILE_trainer(X_a, X_b, model, batch_size = 512, num_epoch=5, 
                        f_temp = 0.1, p_temp = 1.0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = f_temp)
    p_con = ContrastiveLoss(batch_size = model.clf_out,temperature = p_temp)
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    
    for k in range(num_epoch):
        
        model.to(device)
        n = X_a.shape[0]
        r = np.random.permutation(n)
        X_train_a = X_a[r,:]
        X_tensor_A=torch.tensor(X_train_a).float()
        X_train_b = X_b[r,:]
        X_tensor_B=torch.tensor(X_train_b).float()
        
        losses = 0
        
        for j in range(n//batch_size):
            inputs_a = X_tensor_A[j*batch_size:(j+1)*batch_size,:].to(device)
            inputs_a2 = inputs_a + torch.normal(0,1,inputs_a.shape).to(device)
            inputs_a = inputs_a + torch.normal(0,1,inputs_a.shape).to(device)
            
            inputs_b = X_tensor_B[j*batch_size:(j+1)*batch_size,:].to(device)
            inputs_b = inputs_b + torch.normal(0,1,inputs_b.shape).to(device)
            
            feas,o,nfeas,no = model(inputs_a,inputs_b)
            feas2,o2,_,_ = model(inputs_a2,inputs_b)
        
            fea_mi = f_con(feas,nfeas)+f_con(feas,feas2)
            p_mi = p_con(o.T,no.T)+p_con(o.T,o2.T)
        
            #mse_loss = mse(f_a,f_b)
            #pair = torch.ones(f_a.shape[0]).to(device)
            #cos_loss = cos(f_a,f_b,pair)
        
            loss = fea_mi + p_mi #mse_loss + 
            #loss = cos_loss * 0.5 + fea_mi + p_mi
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            losses += loss.data.tolist()
        print("Total loss: "+str(round(losses,4)))
        gc.collect()