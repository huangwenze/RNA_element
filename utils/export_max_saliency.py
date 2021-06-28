from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, argparse, re
import numpy as np
import copy
import hashlib
import torch
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from utils.xsaliency import plot_saliency
from utils.utils import load_testset_txt

from var.loader2 import encoding_seq
import var.net_ics as arch

from var.smoothgrad_main import SaliencyMotif
from utils.checkpoint import make_directory

np.random.seed(247)


def md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()

def mat2str(m):
    string=""
    if len(m.shape)==1:
        for j in range(m.shape[0]):
            string+= "%.3f," % m[j]
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                string+= "%.3f," % m[i,j]
    return string

def decodeDNA(m):
    na=["A","C","G","U"]
    var,inds=np.where(m==1)
    seq=""
    for i in inds:
        seq=seq+na[i]
    return seq

def str_onehot(vec):
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
    
def read_tsv_file(data_file):
    infile = open(data_file,'r')
    corr_dict = {}
    for line in infile:
        #A    ENST00000618697|ENSG00000278022    GTGTCTCTTTGT    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1      -3.12725    0
        line = line.strip('\n')
        sent1 = line.split('\t')
        if (sent1[0] == "A"):
            corr_dict[sent1[2]] = [sent1[1], sent1[2], sent1[3], sent1[4], sent1[5]]
    infile.close()
    return corr_dict

#def output_np_data(data, label_data, out_file):
#    outfile = open(out_file, 'w')
#    #outfile.write('A\t%s|%s|%s|%s\t%s\t%s\t%s\t0\t%s\n' %(ke, gene_id, r-1000, r, seq_dict[ke][r-1000:r], ",".join(shape_dict[ke][r-1000:r]), str(sum), str(sum1)))
#    for i in range(len(data1['seqs'])):
#        if data1['seqs'][i] in label_data:
#            sent = label_data[data1['seqs'][i]]
#            outfile.write('%s\t%s\t%s\t%s\t%s\n' % (sent[0], sent[1], sent[2], sent[3], sent[4]))
#    
#    outfile.close()

def max_per_seq_str(seq_len, data1, slide_len):
    sign_list = data1.max(axis=1)
    #str_sign = []
    str_sign = [sum(sign_list[i:(i+slide_len)]) for i in range(seq_len - slide_len + 1)]
    max_index = str_sign.index(max(str_sign))
    return (max_index)


def main():
    train_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    total_data_tsv = sys.argv[3]
    out_file = sys.argv[4]
    train_data = np.load(train_data_file, allow_pickle=True)
    test_data = np.load(test_data_file, allow_pickle=True)
    total_data = read_tsv_file(total_data_tsv)
    df = pd.read_csv(total_data_tsv, sep='\t')
    outfile = open(out_file, 'w')
    for i in range(train_data['seqs'].shape[0]):
        max_index = max_per_seq_str(len(train_data['seqs'][i]), train_data['saliency'][i], 200)
        Sequence = train_data['seqs'][i]
        Sequence = re.sub("U", "T", Sequence)
        outfile.write('%s\t%d\n' %(total_data[Sequence][0], max_index))
		high_Sequence = Sequence[max_index:(max_index + 200 - 1)]
		
    outfile.close()
        

    
if __name__ == "__main__":
    main()
