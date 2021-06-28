from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, argparse
import numpy as np
import copy
import hashlib
import torch
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from utils.xsaliency import plot_saliency
from utils.utils import load_testset_txt

from var.loader import IcShape, PadSequence
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

def plotmotif(arch_name='ResRNANet', ss_type = 'pu', p_name = 'cds', \
        num=10, gcaug=False,   pos=False, show_plot = True, analysis=0, outfile='out.txt', \
        data_suffix="2", mu=6, test_file="mu.txt", model_suffix="22",hidden_dim=20, \
        datadir = 'data'):
    
    model_dir = "models" + "/" + p_name +"_"+str(hidden_dim)
    model_path = model_dir +"/_"+ss_type+"_1/"+arch_name+".pkl"
    
    device = 0
    

    if ss_type == 'pu':
        nstr = 1
        ss_type = 'pu'
    else:
        nstr = 0
        ss_type = 'seq'
    best_ckpt_file = model_path
    print("loading best model:", best_ckpt_file)

    only_seq=True
    if nstr>0:
        only_seq=False
        
        
    print("Network Arch:", arch_name)
    model = getattr(arch, arch_name)(input_dim=nstr+4, hidden_dim=hidden_dim, output_dim=1, num_layers=2, biFlag=True)
    print(model)
    model.to(device)
    model.eval()

    param = torch.load(best_ckpt_file)
    model.load_state_dict(param)

    test_set = IcShape(datadir, p_name, None, ss_type, is_test=True)
    #test_set = IcShape(datadir, p_name, ss_type, is_test=True, use_npz=True)
    batch_size = 10 #len(test_set)
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    test_loader = torch.utils.data.DataLoader(test_set, \
        batch_size=batch_size, shuffle=False, collate_fn=PadSequence(), **kwargs)
    
    print("Test set:", len(test_loader.dataset))


    results_dir = "results/imgs"
    make_directory("results","imgs")
    out_dir = make_directory(results_dir, p_name)


    saliency = SaliencyMotif(model, only_seq=False, train=False)

    seqs, strs, y_trues, y_preds, saliencys = [],[],[],[],[]
    for j, batch in enumerate(test_loader):
        print("=====================",j)
        X, X_lengths, Y = batch
        if Y[0] ==1:
            IIII=0
        else:
            continue
        X, Y = X.to(device).float(), Y.to(device).float()
        (Y_P, _) = model(batch)
        X_np, Y_np = X.data.cpu().numpy(), Y.data.cpu().numpy()
        Y_P_np = Y_P.data.cpu().numpy()

        predictions = Y_P_np

        is_sort=True
        num_plots = np.minimum(num, batch_size)
        if num_plots == -1:
            num_plots = predictions.shape[0]
        if is_sort:
            max_indices = np.argsort(predictions[:,0])[::-1]
        else:
            max_indices = range(0, predictions.shape[0])
        plot_range = range(0,num_plots)
        plot_index = max_indices[:num_plots]

        X, X_lengths, Y = X[plot_index.tolist()],X_lengths[plot_index.tolist()], Y[plot_index.tolist()]
        X_np_sort, Y_np_sort, Y_P_np_sort = X_np[plot_index], Y_np[plot_index], Y_P_np[plot_index]
        batch = (X, X_lengths, Y)

        for i in plot_range:
            x   = X[i:i+1][:,:]
            x_len=X_lengths[i:i+1][:]
            y   = Y[i:i+1][:]
            min_batch = (x,x_len,y)
            
            guided_saliency = None
            try:
                guided_saliency = saliency.getSmoothSaliency(min_batch)
            except RuntimeError as e:
                print(e)
            if guided_saliency is None:
                continue
            
            new_guided_saliency = x * guided_saliency
            guided_saliency     = guided_saliency.data.cpu().numpy()
            new_guided_saliency = new_guided_saliency.data.cpu().numpy()
            
            x_np, y_np, y_p_np = X_np_sort[i:i+1][:,:], Y_np_sort[i:i+1:], Y_P_np_sort[i:i+1]

            k=0
            m       = x_np[k,:, :4]
            s       = x_np[k,:,-1:]
            y       = y_np[k]
            y_pred  = y_p_np[k]
            print("p, 1min,max:", y_pred, np.min(new_guided_saliency), np.max(new_guided_saliency))

            if y >=1 and y_pred>0.5 :
                iiii=0
            else:
                continue
            seq = decodeDNA(m)
            name_md5 = md5(seq)
            print("sequence: ",len(seq))
            
            if not show_plot:
                n_channel = x.shape[-1] - 1

                str_raw = mat2str(np.squeeze(x[0,:,n_channel]))
                str_sal = mat2str(np.squeeze(new_guided_saliency[0,:,:]))
                j = int(i/4)+1
                k = i%4
                out_dir2 = out_dir +"/"+name_md5+"_"+ss_type+".jpg"
       
                seqs.append(seq)
                strs.append(s)
                y_trues.append(y)
                y_preds.append(y_pred)
                saliencys.append(new_guided_saliency[0,:,:])

                if analysis==0 or analysis== 5 or analysis== 61:
                    continue
     
                plot_saliency(x_np[0],new_guided_saliency[0], guided_saliency[0], i, use_null=4, use_mask=False,use_rescale = True, show_plot=False, outdir=out_dir2)

                continue
            else:
                print("#%d targets: %f pred: %f" %(i, y, y_pred))
                print("%s" %(seq))
                x_np_n = x_np[0] 
                for m in range(0,x_np_n.shape[0],100):
                    outpath = out_dir +"/"+name_md5+"_"+ss_type+"_"+str(m)+".jpg"
                    plot_saliency(x_np[0][m:m+100], new_guided_saliency[0][m:m+100], guided_saliency[0][m:m+100], i, 
                        use_null=4, use_mask=False, use_rescale = True, show_plot=True, outdir=outpath)
    if not show_plot:
        print("totoal true positive samples: ", len(strs))
        np.savez(outfile+".npz", seqs=seqs, icshape=strs, y_true=y_trues, y_pred=y_preds, saliency=saliencys)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch RBP Example')
    # parser.add_argument('--datadir', required=True, help='data name')
    # parser.add_argument('--testset', required=True, help='data path')
    parser.add_argument('--arch', default="Conv5FC3K5_s", help='data path')
    parser.add_argument('--optimizer', default="adam", help='data path')
    parser.add_argument('--mask_func', default="mse", help='data path')
    parser.add_argument('--train_mask', action='store_true', help='data path')
    parser.add_argument('--train_mask2', action='store_true', help='train mask by Hourglass')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--log_interval', type=int, default=50, help='input batch size')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--start', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--beta', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_decay_scale', type=float, default=0.1, help='learning rate, default=0.0002')
    parser.add_argument('--lr_decay_epoch', type=int, default=10, help='learning rate, default=0.0002')
    parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='learning rate, default=0.001')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outf', default='models', help='folder to output images and model checkpoints')
    parser.add_argument('--seed', type=int, default=512, help='manual seed')
    parser.add_argument('--num', type=int, default=10, help='plot n samples ')
    parser.add_argument('--exp_name', type=str, default="cnn", metavar='N',
                        help='experiment name')
    parser.add_argument('--p_name', type=str, default="cds", metavar='N',
                        help='protein name')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--cam', action='store_true', help='cam')
    parser.add_argument('--model_par1', type=str, default="model_origin.pkl", help='input model parameters')
    parser.add_argument('--model_par2', type=str, default="model_param.pkl", help='output model parameter')
    parser.add_argument('--restore_best', action='store_true', help='restore_best')
    parser.add_argument('--eval', action='store_true', help='restore_best')
    parser.add_argument('--seq', action='store_true', help='restore_best')
    parser.add_argument('--out_dir', type=str, default="motif_dir", help='output directory')
    parser.add_argument('--show_seqstr_motif', action='store_true', help='show_seqstr_motif')
    parser.add_argument('--sigmoid', action='store_true', help='sigmoid')
    parser.add_argument('--hidden_dim', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--motif_w', action='store_true', help='motif weight')
    #parser.add_argument('--motif_w', action='store_true', help='motif weight')
    parser.add_argument('--nstr', type=int, default=1, help='number of vector encoding for structure data')
    parser.add_argument('--analysis', type=int, default=0, help='number of vector encoding for structure data')
    parser.add_argument('--ss_type', type=str, default="pu", help='output directory')
    parser.add_argument('--tfboard', action='store_true', help='tf board')

    #parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
    #                    help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.001,
                        help='scale sparse rate (default: 0.0001)')
    args = parser.parse_args()    

    outfile = args.p_name+"_saliency"
    plotmotif(arch_name=args.arch, ss_type = args.ss_type, p_name = args.p_name, \
        num=args.num, gcaug=False, pos=False, show_plot = False, analysis=args.analysis, outfile=outfile, \
        data_suffix="2", mu=6, test_file="mu.txt", model_suffix="22",hidden_dim=args.hidden_dim)

                


if __name__ == "__main__":
    main()