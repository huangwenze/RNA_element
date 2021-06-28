import os, sys
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#from deepomics import utils #, fit, saliency
from utils import visualize

def rescale(vec, thr=0.2):
    ind0 = np.where(vec>=thr)[0]
    u_norm = 0.5 * (vec[ind0]-thr)/(1-thr) + 0.5
    ind1 = np.where(vec<thr)[0]
    p_norm = 0.5 * vec[ind1]/thr
    ind2 = np.where(vec==-1)[0]
    #n_norm = vec[ind2]
    vec_norm = vec.copy()
    vec_norm[ind0] = u_norm
    vec_norm[ind1] = p_norm
    vec_norm[ind2] = 0.5
    return vec_norm

def plot_saliency(X, new_guided_saliency, guided_saliency, i, use_null=4, use_mask=False,\
    use_rescale = True, show_plot=True, outdir="results"):
    if X.shape[-1]==4:
        visualize.plot_seq_saliency(X.T, new_guided_saliency.T, nt_width=100, norm_factor=3, show_plot=show_plot, outdir=outdir)
        return 
   
    
    """
    pu version,
    """

    thr = 0.15
    #print("X.shape:",X.shape)
    x_seq = X[:,:4]
    x_str = X[:,4:]
    s_seq = guided_saliency[:,:4]
    s_str = guided_saliency[:,4:]
    #print("s_str:",s_str.min())

    str_sal = np.concatenate((s_str, s_str),axis=1)
    x_str_null = np.zeros_like(x_str)
    mask_str = np.zeros(str_sal.shape)
    # import pdb; pdb.set_trace()
    ind = np.where(x_str >= thr)[0]
    mask_str[ind,1] = 1
    ind = np.where(x_str < thr)[0]
    mask_str[ind,0] = 1
    ind = np.where(x_str == -1)[0]
    mask_str[ind,0]   = 0.5
    mask_str[ind,1]   = 0.5
    x_str_null[ind,0] = 1
    #new_guided_saliency[i,:,:,4:] = str_sal * mask_str
    new_sal = np.concatenate((new_guided_saliency[:,:], new_guided_saliency[:,4:]), axis=1)
    new_sal[:,4:] = str_sal * mask_str

    #X_6  = np.concatenate((x_seq , 1-x_str, x_str), axis=2)
    if use_rescale:
        x_str_norm = rescale(x_str, thr)
    X_6  = np.concatenate((x_seq , 1-x_str_norm, x_str_norm), axis=1)
    if use_mask:
        X_6[:,4:] = mask_str
    plt = visualize.plot_seq_struct_saliency(X_6.T, new_sal.T, nt_width=100, norm_factor=3, str_null=x_str_null[:,0].T, show_plot=show_plot, outdir=outdir)

