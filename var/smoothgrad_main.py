#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad,Variable

#------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from numpy import linalg as LA


DEBUG = False
class SaliencyMotif(object):
    def __init__(self, model, only_seq=False, train=False):
        self.model    = model
        self.only_seq = only_seq
        self.train    = train

    def getSaliency(self, z):
        index = None
        self.model.train()
        x, x_len, pred_label = z
        x = Variable(x.cuda(), requires_grad=True)

        (output, alpha) = self.model.forward2(x, x_len, pred_label)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        # if self.cuda:
        one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
        # else:
            # one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        grad = x.grad #.data.cpu().numpy()
        #grad = grad[0, :, :, :]

        return grad

    def getSmoothSaliency(self,batch ):
        x_stddev=0.15
        t_stddev=0.15
        nsamples=20
        magnitude=2
        # 1. for sequece
        z, z_len, y = batch
        x = z[:,:,:4] # .data.cpu()
        x_stddev *= (x.max()-x.min()).cuda().item() #.numpy()
        total_grad = torch.zeros(z.shape).cuda()
        x_noise = torch.zeros(x.shape).cuda()
        if not self.only_seq:
            # 2. for structure  
            t = z[:,:,4:] #.data.cpu()
            t_stddev *= (t.max()-t.min()).cuda().item() #.numpy()
            #t_total_grad = torch.zeros(t.shape)
            t_noise = torch.zeros(t.shape).cuda()

        for i in range(nsamples):
            #import pdb;pdb.set_trace()
            x_plus_noise = x + x_noise.zero_().normal_(0, x_stddev)
            if self.only_seq:
                z_plus_noise = x_plus_noise
            else:
                t_plus_noise = t + t_noise.zero_().normal_(0, t_stddev)
                z_plus_noise = torch.cat((x_plus_noise, t_plus_noise), dim=2)
            #print("z_plus_noise:",z_plus_noise.size())
            # grad = self.getMask(z_plus_noise,y)
            grad = self.getSaliency((z_plus_noise,z_len,y))
            
            if magnitude == 0:
                total_grad += grad 
            elif magnitude == 1:
                total_grad += torch.abs(grad)
            elif magnitude == 2:
                total_grad += grad * grad
                #import pdb; pdb.set_trace()
            

            #total_grad += grad * grad
        total_grad /= nsamples
        return total_grad


