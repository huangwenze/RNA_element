from __future__ import print_function
import os, sys
import numpy as np
import hashlib

import torch
import pandas as pd
from sklearn import metrics


__all__ = [
	"placeholder",
	"Variable",
	"make_directory",
	"normalize_pwm",
	"meme_generate",
    "md5",
    "mat_to_str",
    "decode_DNA",
    "tensor_to_DNA",
    "decode_RNA",
    "icSHAPE_to_onehot",
]


def md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()

def mat_to_string(m):
    string=""
    if len(m.shape)==1:
        for j in range(m.shape[0]):
            string+= "%.3f," % m[j]
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                string+= "%.3f," % m[i,j]
    return string

def encode_DNA(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""

    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq

def decode_DNA(m):
    na=["A","C","G","U"]
    var,inds=np.where(m==1)
    seq=""
    for i in inds:
        seq=seq+na[i]
    return seq

def tensor_to_DNA(t):
    x_np = t.data.cpu().numpy()
    m  = x_np[0,0,:, :4]
    return decode_DNA(m)

def decode_RNA(m):
    #na=["A","C","G","U"]
    #cl=['g','b','orange','r']
    #if type=="AUCG":
    na=["A","U","C","G"]
    cl=['g','r','b','orange']
    var,inds=np.where(m==1)
    #print(inds)
    seq=""
    for i in inds:
        seq=seq+na[i]
    return seq

def icSHAPE_to_onehot(vec):
    thr=0.15
    mask_str = np.zeros((2,vec.shape[-1]))
    ind =np.where(vec >= thr)[1]
    mask_str[1,ind]=1
    ind =np.where(vec < thr)[1]
    mask_str[0,ind]=1
    ind =np.where(vec == -1)[1]
    mask_str[0,ind]=0.5
    mask_str[1,ind]=0.5
    return mask_str

def normalize_pwm(pwm, factor=None):

	MAX = np.max(np.abs(pwm))
	pwm = pwm/MAX
	if factor:
		pwm = np.exp(pwm*factor)
	norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
	return pwm/norm



def acc_auc(output,  y):
    y1      = y.to(device='cpu', dtype=torch.long).numpy()
    p_class = (output>=0.5).to(device='cpu').data.numpy()
    prob   = output.to(device='cpu').data.numpy()
    acc = metrics.accuracy_score(y1, p_class)
    auc = 0.5
    try:
        auc = metrics.roc_auc_score(y1, prob)
    except Exception as e:
        pass
    
    return acc, auc

def cal_acc_auc(loader, model, device, dtype):
    with torch.no_grad():
        acc_lst = []
        auc_lst = []
        for i, (x0, t0, r0, b0, y0) in enumerate(loader):
            X=torch.cat((x0,t0),dim=3)
            X = X.to(device=device, dtype=dtype)
            y = y0.to(device='cpu', dtype=torch.long)[:,1]
            scores = model(X)
            _, predict_labels = scores.max(dim=1)
            predict_labels = predict_labels.to(device='cpu', dtype=torch.long)
            acc_lst.append(metrics.accuracy_score(y, predict_labels))
            auc_lst.append(metrics.roc_auc_score(y, predict_labels))
        acc = np.mean(acc_lst)
        auc = np.mean(auc_lst)
    return acc, auc

def clip_gradient_model(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    #import pdb
    #pdb.set_trace()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def load_testset_txt(filepath, test):
    print("Reading test file:", filepath)
    max_length=101
    f_mu = open(filepath,"r")
    seqs = []
    strs = []
    use_pu = True
    if test['inputs'].shape[-1]==4:
        use_pu = False
    for line in f_mu.readlines():
        line=line.strip('\n').split('\t')
        seqs.append(line[2])
        if use_pu:
            strs.append(line[3])
    in_seq = convert_one_hot(seqs, max_length)
    in_ver = 5
    if "7v" in filepath:
        in_ver = 7
    if use_pu:
        structure = np.zeros((len(seqs), in_ver-4, max_length))
        for i in range(len(seqs)):
            icshape = strs[i].strip(',').split(',')
            #print(icshape)
            ti = [float(t) for t in icshape]
            ti = np.array(ti).reshape(1,-1)
            if in_ver == 5:
                pu = np.concatenate([ti], axis=0)
            elif in_ver == 6:
                pu = np.concatenate([1-ti, ti], axis=0)
            elif in_ver == 7:
                pu = str_onehot(ti)
                pu = np.concatenate([pu, ti], axis=0)
            #in_str = np.concatenate([ti], axis=0)
            structure[i]=pu
        #print("in_seq",in_seq.shape)
        #print("in_str",structure.shape)
        input = np.concatenate([in_seq, structure], axis=1)
    else:
        input = in_seq

    inputs = np.expand_dims(input, axis=3).transpose([0, 2, 3, 1])
    targets = np.ones((in_seq.shape[0],1))

    #targets = np.ones((in_seq.shape[0]+1,1))
    targets[in_seq.shape[0]-1]=0

    #inputs = np.concatenate([inputs, inputs[:1,:,:,:]], axis=0)

    test['inputs'] =inputs
    test['targets'] =targets
    return test
