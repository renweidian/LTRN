# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:32:24 2020

@author: Dian
"""
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import misc
import cv2
from scipy.io import loadmat
import sys
from matplotlib.pyplot import *
from numpy import *
from torch.nn import functional as F
from torch import nn
import torch
import os

import torch.utils.data as data
import metrics
import hdf5storage
import h5py



class HSI_MSI_Data3(data.Dataset):
    def __init__(self,path,R,training_size,stride,num,data_name='cave'):
        if data_name=='cave':
            save_data(path=path,R=R,training_size=training_size,stride=stride,num=num,data_name=data_name)
            data_path = 'E:\super-resolution\spectraldata\cave_patchdata\\'
        elif data_name == 'Harved':
            save_data(path=path,R=R,training_size=training_size,stride=stride,num=num,data_name=data_name)
            data_path = 'E:\super-resolution\spectraldata\Harved_patchdata\\'
        else:
            raise Exception("Invalid mode!", data_name)
        imglist=os.listdir(data_path)
        self.keys = imglist
        self.keys.sort()
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
#        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
#        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

    
            

def reconstruction(net2,R,R_inv,MSI,training_size,stride):
        index_matrix=torch.zeros((R.shape[1],MSI.shape[2],MSI.shape[3])).cuda()
        abundance_t=torch.zeros((R.shape[1],MSI.shape[2],MSI.shape[3])).cuda()
        a=[]
        for j in range(0, MSI.shape[2]-training_size+1, stride):
            a.append(j)
        a.append(MSI.shape[2]-training_size)
        b=[]
        for j in range(0, MSI.shape[3]-training_size+1, stride):
            b.append(j)
        b.append(MSI.shape[3]-training_size)
        for j in a:
            for k in b:
                temp_hrms = MSI[:,:,j:j+training_size, k:k+training_size]
#                temp_hrms=torch.unsqueeze(temp_hrms, 0)
#                print(temp_hrms.shape)
                with torch.no_grad():
                    # print(temp_hrms.shape)
#                    HSI = net2(R,R_inv,temp_hrms)
                    HSI = net2(temp_hrms)
                    HSI=HSI.squeeze()
#                   print(HSI.shape)
                    HSI=torch.clamp(HSI,0,1)
                    abundance_t[:,j:j+training_size, k:k+training_size]= abundance_t[:,j:j+training_size, k:k+training_size]+ HSI
                    index_matrix[:,j:j+training_size, k:k+training_size]= 1+index_matrix[:,j:j+training_size, k:k+training_size]
                
        HSI_recon=abundance_t/index_matrix
        return HSI_recon     
def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F
class HSI_MSI_Data(data.Dataset):
    def __init__(self,train_hrhs_all,train_hrms_all):
        self.train_hrhs_all  = train_hrhs_all
        self.train_hrms_all  = train_hrms_all
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms
    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data1(data.Dataset):
    def __init__(self,path,R,training_size,stride,num):
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            img1=img1/img1.max()
            HRHSI=np.transpose(img1,(2,0,1))
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = train_hrhs
         self.train_hrms_all  = train_hrms
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data2(data.Dataset): 
    def __init__(self,path,R,training_size,stride,num):      
         imglist=os.listdir(path)
         train_hrhs=[]
         # train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=[]
         # train_hrms=torch.Tensor(train_hrms)
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["ref"]
            img1=img1/img1.max()
#            HRHSI=np.transpose(img1,(2,0,1))
#            MSI=np.tensordot(R, HRHSI, axes=([1], [0]))
            HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
            MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
            HRHSI = HRHSI.numpy()
            MSI = MSI.numpy()
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
#         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = torch.Tensor(train_hrhs)
         self.train_hrms_all  = torch.Tensor(train_hrms)
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]



def dataacquire(path,R,training_size,stride,num):      
  imglist=os.listdir(path)
  train_hrhs=[]
  train_hrms=[]
  for i in range(num):
    img=loadmat(path+imglist[i])
    img1=img["ref"]
    img1=img1/img1.max()
    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
    HRHSI = HRHSI.numpy()
    MSI = MSI.numpy()
    for j in range(0, HRHSI.shape[1]-training_size+1, stride):
        for k in range(0, HRHSI.shape[2]-training_size+1, stride):
            temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
            temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
            train_hrhs.append(temp_hrhs)
            train_hrms.append(temp_hrms)
  return train_hrhs,train_hrms
     






def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr=init_lr1+iteraion/warm_iter*(init_lr2-init_lr1)
    else:
      lr = init_lr2*(1 - (iteraion-warm_iter)/(max_iter-warm_iter))**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F

class LossTrainCSS(nn.Module):
    def __init__(self):
        super(LossTrainCSS, self).__init__()
       

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / (label+1e-10)
        mrae = torch.mean(error)
        return mrae
#
#    def mrae_loss(self, outputs, label):
#        error = torch.abs(outputs - label) / label
#        mrae = torch.mean(error)
#        return mrae

    def rgb_mrae_loss(self, outputs, label):
        error = torch.abs(outputs - label)
        mrae = torch.mean(error.view(-1))
        return mrae
 
    
class MyarcLoss(torch.nn.Module):
    def __init__(self):
        super(MyarcLoss, self).__init__()

    def forward(self, output, target):
        sum1=output*target
        sum2=torch.sum(sum1,dim=0)+1e-10
        norm_abs1=torch.sqrt(torch.sum(output*output,dim=0))+1e-10
        norm_abs2=torch.sqrt(torch.sum(target*target,dim=0))+1e-10
        aa=sum2/norm_abs1/norm_abs2
        aa[aa<-1]=-1
        aa[aa>1]=1
        spectralmap=torch.acos(aa)
        return torch.mean(spectralmap)
     
     
     
class AWCA(nn.Module):
    def __init__(self, channel=31):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, 1, bias=False),
#            nn.PReLU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h*w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Projection_R(nn.Module):
    def __init__(self):
        super(Projection_R, self).__init__()
    def forward(self,R,MSI, mu,Y):
        RTR=torch.mm(torch.transpose(R, 1, 0),R)
        x = torch.tensordot(MSI, R, dims=([1], [0])) 
        x=torch.Tensor.permute(x,(0,3,1,2))+mu*Y
        x=torch.tensordot(x, torch.inverse(RTR+mu*torch.eye(R.shape[1]).cuda()), dims=([1], [1])) 
        x=torch.Tensor.permute(x,(0,3,1,2))
        return x  

      
class Spectral_Pro(nn.Module):
    def __init__(self,in_chanels,out_chanels):
        super(Spectral_Pro, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(in_chanels,out_chanels, 3, 1, 1,bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            AWCA(out_chanels),
            nn.Conv2d(out_chanels, 31, 3, 1, 1,bias=True),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ) 
        
#       self.non_local=AWAN1.PSNL(31)
#        self.down_relu=nn.LeakyReLU(negative_slope=0.2, inplace=False)
    def spectral_projection(self,R,MSI, mu,Y):
        RTR=torch.mm(torch.transpose(R, 1, 0),R)
        x = torch.tensordot(MSI, R, dims=([1], [0])) 
        x=torch.Tensor.permute(x,(0,3,1,2))+mu*Y
        x=torch.tensordot(x, torch.inverse(RTR+mu*torch.eye(R.shape[1]).cuda()), dims=([1], [1])) 
        x=torch.Tensor.permute(x,(0,3,1,2))
        return x  
    def forward(self,R, x,MSI,mu):
        x = self.conv1(x)
#        x=self.non_local(x)
        x = self.spectral_projection(R,MSI, mu,x)
        return x




class CNN_BP_SE2(nn.Module):
    def __init__(self,mu,num=128):
        super(CNN_BP_SE2, self).__init__()
        self.mu=mu
        self.pro=Spectral_Pro(3,num)
        self.pro1=Spectral_Pro(31,num)
        self.pro2=Spectral_Pro(31,num)
        self.pro3=Spectral_Pro(31,num)
        self.pro4=Spectral_Pro(31,num)
        self.pro5=Spectral_Pro(31,num)
        self.pro6=Spectral_Pro(31,num)
        self.pro7=Spectral_Pro(31,num)
    def forward(self, R,R_inv, MSI):
        mu=self.mu
#        x=torch.tensordot(MSI, R_inv, dims=([1], [1])) 
#        x=torch.Tensor.permute(x,(0,3,1,2))
#        b=torch.cat((MSI,x),1)
        x=self.pro(R,MSI, MSI,mu)
        x=self.pro1(R,x, MSI,mu)
        x=self.pro2(R,x, MSI,mu)
        x=self.pro3(R,x, MSI,mu)
        x=self.pro4(R,x, MSI,mu)
        x=self.pro5(R,x, MSI,mu)
        x=self.pro6(R,x, MSI,mu)
        return x


class CNN_BP_SE3(nn.Module):
    def __init__(self,mu,n_DRBs=4):
        super(CNN_BP_SE3, self).__init__()
        self.mu=mu
        num=128
        self.pro=Spectral_Pro(3,num)
        self.pro1=Spectral_Pro(31,num)
        self.pro2=Spectral_Pro(31*2,num)
        self.pro3=Spectral_Pro(31*3,num)
        self.pro4=Spectral_Pro(31*4,num)
        self.pro5=Spectral_Pro(31*5,num)
        self.pro6=Spectral_Pro(31*6,num)
        self.pro7=Spectral_Pro(31*7,64)
        self.pro8=Spectral_Pro(31*8,64) 

    def forward(self, R,R_inv, MSI):
        mu=self.mu
#        x=torch.tensordot(MSI, R_inv, dims=([1], [1])) 
#        x=torch.Tensor.permute(x,(0,3,1,2))
#        x1=torch.cat((MSI,x),1)
        x1=MSI
        x=self.pro(R,x1, MSI,mu)
        x1=x
        x=self.pro1(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro2(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro3(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro4(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro5(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro6(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro7(R,x1, MSI,mu)
        x1=torch.cat((x1,x),1)
        x=self.pro8(R,x1, MSI,mu)
        return x



class RTGB(nn.Module):
    def __init__(self, channel, width, height):
        super(RTGB, self).__init__()
        self.c_rtgb = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel, (1, 1)),       # 不能padding
            nn.Sigmoid()
        )
        self.w_rtgb = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(width, width, (1, 1)),
            nn.Sigmoid()
        )
        self.h_rtgb = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(height, height, (1, 1)),
            nn.Sigmoid()
        )
        # print('实例化')

    def forward(self, x):
        #print("之前的B,C,W,H:{}".format(x.shape))

        c_1 = self.c_rtgb(x)  # 计算w*h面上的均值，是一个1*C的向量
        #print("B,C,W,H:{}".format(x.shape))

        x = x.permute(0, 2, 1, 3)  # 把宽度放到第一个维度，对C*H进行卷积
        #print("B,W,C,H:{}".format(x.shape))
        w_1 = self.w_rtgb(x)  # 计算C*H面上的均值，是一个1*W的向量
        x = x.permute(0, 3, 1, 2)  # 把H轴放到第一个维度，对W*C进行卷积
        #print("B,H,W,C:{}".format(x.shape))

        h_1 = self.h_rtgb(x)  # 计算W*C面上的均值，是一个1*H的向量
        cb,cc,cw,ch = c_1.shape     
        wb,wc,ww,wh = w_1.shape
        hb,hc,hw,hh = h_1.shape     # 第一个是批量，第二个是通道数,批量都是一样的
        #print("c_1：{}".format(c_1.shape))
        #print("w_1：{}".format(w_1.shape))
        #print("h_1：{}".format(h_1.shape))
        matrix3d = torch.empty(cb, wc,hc).cuda()  
        for i in range(0,cb-1):    
            matrix3d[i] = torch.outer(w_1.view(cb,-1)[i], h_1.view(cb,-1)[i])  # Kronecker product得到bachsize个W*H的矩阵
        #print("应该是带batch的三维的张量的{}".format(matrix3d.shape))
        matrix4d = torch.empty(cb, cc,wc,hc).cuda()           
        for i in range(0,cb-1):
            #print("一个c_1:{}".format(c_1.view(cb,-1)[i].shape))
            #print("一个matrix3d[i]:{}:".format(matrix3d[i].view(-1).shape))
            matrix4d[i] = torch.outer(c_1.view(cb,-1)[i], matrix3d[i].view(-1)).view(cc,wc,hc)  # Kronecker product积操作，得到product得到bachsize个W个C*(W*H)的二维张量
        #print("应该是batch的四维的张量的{}".format(matrix4d.shape))
        # output = max4dx.unsqueeze(0).view(cb,cc, wc, hc)  # 将张量重新变形为C*W*H的三位张量，ouput就是秩1张量
        output = matrix4d 
        return output


class DRTLM(nn.Module):
    def __init__(self, channel, width, height):
        super(DRTLM, self).__init__()
        self.rtgb1 = RTGB(channel, width, height)
        self.rtgb2 = RTGB(channel, width, height)
        self.rtgb3 = RTGB(channel, width, height)
        self.rtgb4 = RTGB(channel, width, height)
        self.conv = nn.Conv2d(channel * 5, channel, (3, 3),padding=1)   # 因为要进行Hadamard product，所示与输入时的通道一致

    def forward(self, x):
        o1 = self.rtgb1(x)  # x是C*W*H的  o1也是    o2也是
        o2 = self.rtgb2(x - o1)
        o3 = self.rtgb3(x - o1 - o2)
        o4 = self.rtgb4(x - o1 - o2 - o3)
        o5 = self.rtgb4(x - o1 - o2 - o3 - o4)
        # print(o1.shape)
# o1+o2这种是后来加的
        o = torch.cat((o1, o1+o2, o1+o2+o3, o1+o2+o3+o4, o1+o2+o3+o4+o5), dim=1)
        # print("秩1拼接后的张量：{}".format(o.shape))
        output = self.conv(o)                       # 学习CP分解的系数
        return output
    
class CNN_CP(nn.Module):
    def __init__(self, in_channel,cpin_channel,out_channel, width, height):
        super(CNN_CP,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,64,(3,3),padding=1)
        self.conv2 = nn.Sequential(                            # 特征提取？卷积+relu   编码？
                nn.Conv2d(64,128,(3,3),padding=1),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                nn.Conv2d(128,cpin_channel,(3,3),padding=1)
                )
        self.drtlm = DRTLM( cpin_channel, width, height)    #CP分解、
#尝试0       好像是batchsize=16时  4.16   8.43    同时第一步的激活函数为ReLU
#        self.conv2= nn.Sequential(                                            
#                nn.Conv2d(cpin_channel,128,(3,3),padding=1),
#                nn.Conv2d(128,out_channel,(3,3),padding=1)                
#                )
#尝试1       batchsize=64时，结果不好
#        self.conv2= nn.Sequential(                                            
#                nn.Conv2d(cpin_channel,128,(3,3),padding=1),
#                nn.LeakyReLU(negative_slope=0.2, inplace=False),
#                nn.Conv2d(128,out_channel,(3,3),padding=1)                
#                )
#尝试2  设置batchsize=64  200 EPOCH ，    5.9677   10.2196    下降很慢
#      设置batchsize=32  200 EPOCH       4.38  8.09           
#      设置batchsize=16  100 EPOCH       4.267254811038313   7.8634744          
      
#        self.conv2= nn.Sequential(                                              # 
#                nn.Conv2d(cpin_channel,128,(3,3),padding=1),
#                nn.Conv2d(128,256,(3,3),padding=1) ,
#                nn.LeakyReLU(negative_slope=0.2, inplace=False),
#                nn.Conv2d(256,128,(3,3),padding=1) ,
#                nn.Conv2d(128,out_channel,(3,3),padding=1)                
#                )
#尝试3           
#      设置batchsize=16  200 EPOCH     4.388070108591077 7.847916            
#      
#        self.conv2= nn.Sequential(                                              # 
#                nn.Conv2d(cpin_channel,128,(3,3),padding=1),
#                nn.LeakyReLU(negative_slope=0.2, inplace=False),
#                nn.Conv2d(128,out_channel,(3,3),padding=1)                
#                )
#尝试4
        self.conv3= nn.Sequential(                                                # 尝试残差
                nn.Conv2d(cpin_channel,128,(3,3),padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=False),                
                nn.Conv2d(128,128,(3,3),padding=1) ,
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                nn.Conv2d(128,128,(3,3),padding=1) ,
                AWCA(128),         
                nn.Conv2d(128,cpin_channel,(3,3),padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=False),                
                )
        self.conv4= nn.Sequential(
                nn.Conv2d(cpin_channel,128,(3,3),padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                nn.Conv2d(128,out_channel,(3,3),padding=1) ,
                )

        
    def forward(self,x):
        '''
        try:
            print("原始的大小：{}".format(x.shape))
        except:
            print(x)
        '''
        x1=self.conv1(x) 
        x2 = self.conv2(x1)                     # 特征提取
        #print("特征提取后的大小：{}".format(x1.shape))
        
        x3 = self.drtlm(x2)  
        #print("x2:{}".format(x2.shape))        # 做CP分解
        #print("x1:{}".format(x1.shape))
        x4 = x3*x2                             # 做Hadamard product    还是cpin_channel的通道数
        #print("x3:{}".format(x3.shape))
# +x0是后来加的
        x5 = self.conv3(x4+x1)           # 后续进行卷积训练，重建,输出31个通道的图像       #残差时改为cpin_channel的通道数        
#普通的
#       return x4
#残差1 
        x6 = self.conv4(x4+x1+x5)
        
        return x6
#残差2        
#        x5 = self.conv2(x3+s4) 
#        x6 = self.conv2(x4+s5) 
#        output = self.conv3(x6)
#        
#        
#        return output        
        
        