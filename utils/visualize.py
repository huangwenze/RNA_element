import os, sys
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.misc import imresize
#from scipy.special import softmax

import pandas as pd
from matplotlib import gridspec
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

import hashlib
import cv2

def md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()

def decodeDNA(m):
    na=["A","C","G","U"]
    var,inds=np.where(m==1)
    #print(inds)
    seq=""
    for i in inds:
        seq=seq+na[i]
    return seq
    
def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm


def plot_roc_all(final_roc):
    """Plot ROC curve for each class"""

    fig = plt.figure()
    for i in range(len(final_roc)):
        plt.plot(final_roc[i][0],final_roc[i][1])
    plt.xlabel('False positive rate', fontsize=22)
    plt.ylabel('True positive rate', fontsize=22)
    plt.plot([0, 1],[0, 1],'k--')
    ax = plt.gca()
    ax.xaxis.label.set_fontsize(17)
    ax.yaxis.label.set_fontsize(17)
    map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
    map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
    plt.tight_layout()
    #plt.legend(loc='best', frameon=False, fontsize=14)
    return fig, plt


def plot_pr_all(final_pr):
    """Plot PR curve for each class"""

    fig = plt.figure()
    for i in range(len(final_pr)):
        plt.plot(final_pr[i][0],final_pr[i][1])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    ax = plt.gca()
    ax.xaxis.label.set_fontsize(17)
    ax.yaxis.label.set_fontsize(17)
    map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
    map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
    plt.tight_layout()
    #plt.legend(loc='best', frameon=False, fontsize=14)
    return fig, plt


def activation_pwm(fmap, X, threshold, window):

    # find regions above threshold
    x, y = np.where(fmap > threshold)

    # sort score
    index = np.argsort(fmap[x,y])[-1:0:-1]
    data_index = x[index].astype(int)
    pos_index = y[index].astype(int)
    activation = fmap[data_index, pos_index]

    # extract sequences with aligned activation
    seq_align = []
    window = int(window/2)
    num_dims = X.shape[2]
    count_matrix = np.zeros((window*2, num_dims))

    for i in range(len(pos_index)):

        start_window = pos_index[i] - window
        if start_window < 0:
            start_buffer = np.zeros((-start_window, num_dims))
            start = 0
        else:
            start = start_window

        end_window = pos_index[i] + window
        end_remainder = end_window - fmap.shape[1]
        if end_remainder > 0:
            end = fmap.shape[1]
            end_buffer = np.zeros((end_remainder, num_dims))
        else:
            end = end_windows

        seq = X[data_index[i], start:end, :]*activation[i]
        counts = np.ones(seq.shape)*activation[i]

        if start_window < 0:
            seq = np.vstack([start_buffer, seq])
            counts = np.vstack([start_buffer, counts])
        if end_remainder > 0:
            seq = np.vstack([seq, end_buffer])
            counts = np.vstack([counts, end_buffer])

        seq_align.append(seq)
        count_matrix += counts
    seq_align = np.array(seq_align)

    seq_align = np.sum(seq_align, axis=0)/count_matrix
    seq_align[np.isnan(seq_align)] = 0

    return seq_align

def generate_pwm(sess, nntrainer, X, guided_saliency, window=6, layer='conv1d_0_active'):

    data={'inputs': guided_saliency}
    fmaps = nntrainer.get_activations(sess, data, layer=layer)

    num_filters = fmaps.shape[-1]

    pwm = []
    for i in range(num_filters):
        fmap = np.squeeze(fmaps[:,:,:,i])

        # get threshold
        threshold = np.max(fmap)*0.8

        pwm.append(activation_pwm(fmap, X, threshold, window))

    return np.array(pwm)

def filter_heatmap(W, norm=True, cmap='hot_r', cbar_norm=True):
    import matplotlib
    if norm:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    else:
        norm = None
    cmap_reversed = matplotlib.cm.get_cmap(cmap)
    im = plt.imshow(W, cmap=cmap_reversed, norm=norm)

    #plt.axis('off');
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, W.shape[1], 1.), minor=True);
    ax.set_yticks(np.arange(-.5, W.shape[0], 1.), minor=True);
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    plt.xticks([]);
    if W.shape[0] == 4:
        plt.yticks([0, 1, 2, 3], ['A', 'C', 'G', 'U'], fontsize=16)
    else:
        plt.yticks([0, 1, 2, 3, 4, 5], ['A', 'C', 'G', 'U', 'paired', 'unpaired'], fontsize=16)

    #cbar = plt.colorbar();
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    if cbar_norm:
        cbar.set_ticks([0.0, 0.5, 1.0])
    return plt


def plot_filter_logos(W, figsize=(10,7), height=25, nt_width=10, norm=0, alphabet='dna', norm_factor=3, sort=True, filepath="kernel.jpg"):

    W = np.squeeze(W.transpose([3, 2, 0, 1]))
    num_filters = W.shape[0]
    num_rows = int(np.ceil(np.sqrt(num_filters)))
    grid = mpl.gridspec.GridSpec(num_rows, num_rows)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)

    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize);
    MAX = np.max(np.abs(W))
    ids = range(num_filters)
    b=[]
    b1=[]
    #c=[]
    if sort:
        for i in range(num_filters):
            b.append(-np.linalg.norm(W[i], ord=2))
            b1.append(np.linalg.norm(W[i], ord=1))
        ids = np.array(b).argsort()
        #print(b[ids])
    j =0
    for i in ids:
        plt.subplot(grid[j]);
        if norm_factor:
            W_norm = normalize_pwm(W[i], factor=norm_factor, MAX=MAX)
        else:
            W_norm = W[i]
        logo = seq_logo(W_norm, height=height, nt_width=nt_width, norm=norm, alphabet=alphabet)
        plot_seq_logo(logo, nt_width=nt_width, step_multiple=None)
        #pwm_struct_saliency = normalize_pwm(struct_saliency, factor=norm_factor)
        #pwm_struct_saliency_logo = seq_logo_reverse(pwm_struct_saliency, height=int(nt_width*8), nt_width=nt_width, norm=0, alphabet='pu', colormap='bw')
        #if np.mod(i, num_rows) != 0:
        title = 'L2: %.3f L1: %.3f' % (-b[i], b1[i])
        plt.title(title, fontsize='large')
        plt.yticks([])
        j+=1
    #save_logo(logo_img, filepath):
    #plt.imshow(logo_img)

    fig.savefig(filepath, format='jpg', dpi=200, bbox_inches='tight')
    return fig, plt

def seqstr_logo_saliency(seqstr, mask, show=True, seq=None):
    seq_sal = seqstr[:,:4]
    mask = mask.numpy()
    size_upsample = (400, 40)
    size_upsample2 = (400, 10)
    size = size_upsample
    str_img = None
    n_img = 2
    if seqstr.shape[1]>=5:
        n_img = 3
        str_sal = seqstr[:,4:]
        str_img = mat2img(str_sal, (400,10*str_sal.shape[1]))

    seq_img = mat2img(seq_sal, size_upsample)
    lba_img = mat2img(mask, (400, 6), rgb=False, rescale=False)
    lba_img = np.tile(lba_img,(3,1,1)).transpose(1,2,0)

    if show:
        #import matplotlib.gridspec as gridspec
        plt.figure(figsize=(20, seqstr.shape[1]),)
        ax1 = plt.subplot(n_img, 1, 1)
        ax2 = plt.subplot(n_img, 1, 2)


        ax1.imshow(lba_img)
        if seq is not None:
            bin = size[0] / seq_sal.shape[0]
            seq_list = list(seq)
            ax1.set_xticklabels(seq_list)
            ax1.set_xticks(np.arange(0, size[0], bin))
        ax1.set_yticks([])

        ax2.imshow(seq_img)
        #ax2.set_xticks([])
        if seq is not None:
            bin = size[0] / seq_sal.shape[0]
            seq_list = list(seq)
            ax2.set_xticklabels(seq_list)
            ax2.set_xticks(np.arange(0, size[0], bin))
        ax2.set_yticklabels(("A","U","C","G"))
        ax2.set_yticks(list(range(0,40,10)))
        if str_img is not None:
            ax3 = plt.subplot(3, 1, 3)
            ax3.imshow(str_img)
            #ax3.set_xticks([])
            if seq is not None:
                bin = size[0] / seq_sal.shape[0]
                seq_list = list(seq)
                ax3.set_xticklabels(seq_list)
                ax3.set_xticks(np.arange(0, size[0], bin))
            if str_sal.shape[1]==1:
                ax3.set_yticklabels(("P"))
            elif str_sal.shape[1]==2:
                ax3.set_yticklabels(("P","U"))
                ax3.set_yticks(list(range(0,20,10)))
            elif str_sal.shape[1]==4:
                ax3.set_yticklabels(("P","H","M","E"))
                ax3.set_yticks(list(range(0,40,10)))


        plt.show()
        #cv2.imwrite(filename, result)
    return seq_img, str_img, lba_img



def mat2img(mat, size, rgb=True, rescale=True):
    """
    colormap: https://docs.opencv.org/2.4/modules/contrib/doc/facerec/colormaps.html
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
    """
    if rescale:
        cam = mat - np.min(mat)
        cam_img = cam / (np.max(mat) - np.min(mat))
    else:
        cam_img = mat
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img.T, size)

    if rgb:
        cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    return cam_img

def seqstr_logo_saliency(seqstr, mask, show=True, seq=None):
    seq_sal = seqstr[:,:4]
    mask = mask.numpy()
    size_upsample = (400, 40)
    size_upsample2 = (400, 10)
    size = size_upsample
    str_img = None
    n_img = 2
    if seqstr.shape[1]>=5:
        n_img = 3
        str_sal = seqstr[:,4:]
        str_img = mat2img(str_sal, (400,10*str_sal.shape[1]))

    seq_img = mat2img(seq_sal, size_upsample)
    lba_img = mat2img(mask, (400, 6), rgb=False, rescale=False)
    lba_img = np.tile(lba_img,(3,1,1)).transpose(1,2,0)

    if show:
        #import matplotlib.gridspec as gridspec
        plt.figure(figsize=(20, seqstr.shape[1]),)
        ax1 = plt.subplot(n_img, 1, 1)
        ax2 = plt.subplot(n_img, 1, 2)


        ax1.imshow(lba_img)
        if seq is not None:
            bin = size[0] / seq_sal.shape[0]
            seq_list = list(seq)
            ax1.set_xticklabels(seq_list)
            ax1.set_xticks(np.arange(0, size[0], bin))
        ax1.set_yticks([])

        ax2.imshow(seq_img)
        #ax2.set_xticks([])
        if seq is not None:
            bin = size[0] / seq_sal.shape[0]
            seq_list = list(seq)
            ax2.set_xticklabels(seq_list)
            ax2.set_xticks(np.arange(0, sizlee[0], bin))
        ax2.set_yticklabels(("A","U","C","G"))
        ax2.set_yticks(list(range(0,40,10)))
        if str_img is not None:
            ax3 = plt.subplot(3, 1, 3)
            ax3.imshow(str_img)
            #ax3.set_xticks([])
            if seq is not None:
                bin = size[0] / seq_sal.shape[0]
                seq_list = list(seq)
                ax3.set_xticklabels(seq_list)
                ax3.set_xticks(np.arange(0, size[0], bin))
            if str_sal.shape[1]==1:
                ax3.set_yticklabels(("P"))
            elif str_sal.shape[1]==2:
                ax3.set_yticklabels(("P","U"))
                ax3.set_yticks(list(range(0,20,10)))
            elif str_sal.shape[1]==4:
                ax3.set_yticklabels(("P","H","M","E"))
                ax3.set_yticks(list(range(0,40,10)))


        plt.show()
        #cv2.imwrite(filename, result)
    return seq_img, str_img, lba_img


def plot_filter_logos2(W, figsize=(10,7), height=25, nt_width=10, norm=0, alphabet='dna', norm_factor=3, sort=True, filepath="kernel.jpg"):
    # 12x1x6x8
    W = np.squeeze(W.transpose([3, 2, 0, 1]))
    # 8x6x12x1
    num_filters = W.shape[0]
    num_rows = int(np.ceil(np.sqrt(num_filters)))
    grid = mpl.gridspec.GridSpec(num_rows, num_rows)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)

    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize);
    MAX_seq = np.max(np.abs(W[:,:4,:]))
    MAX_str = np.max(np.abs(W[:,4:,:]))
    ids = range(num_filters)
    b=[]
    b1=[]
    #c=[]
    if sort:
        for i in range(num_filters):
            b.append(-np.linalg.norm(W[i], ord=2))
            b1.append(np.linalg.norm(W[i], ord=1))
        ids = np.array(b).argsort()
        #print(b[ids])
    j =0
    for i in ids:
        plt.subplot(grid[j]);
        if norm_factor:
            W_seq = normalize_pwm(W[i,:4,:], factor=norm_factor, MAX=MAX_seq)
            W_str = normalize_pwm(W[i,4:,:], factor=norm_factor, MAX=MAX_str)
        else:
            #W_seq = W[i,:4,:]
            #W_str = W[i,4:,:]
            print("W_seq 0:",W[i,:4,:])
            W_seq = softmax(W[i,:4,:], axis=0)
            W_str = softmax(W[i,4:,:], axis=0)
            print("W_seq 1:",W_seq)
            print(W_seq.max())

        logo_seq = seq_logo(W_seq, height=height, nt_width=nt_width, norm=norm, alphabet=alphabet)
        #logo_str = seq_logo_reverse(W_str, height=height, nt_width=nt_width, norm=norm, alphabet='pu', colormap='bw')
        logo_str = mat2img(W_str.T, (120,20))
        spacer = np.zeros([1, W_str.shape[-1]*nt_width, 3], dtype=np.uint8)
        #spacer.fill(255)

        # build logo image
        logo_img = np.vstack([logo_seq, spacer,logo_str])

        title = 'L2: %.3f L1: %.3f' % (-b[i], b1[i])
        plt.title(title, fontsize='large')
        plt.yticks([])
        plt.xticks([])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

        plt.imshow(logo_img, interpolation='none')
        #plot_seq_logo(logo, nt_width=nt_width, step_multiple=None)


        #if np.mod(i, num_rows) != 0:

        j+=1
    #save_logo(logo_img, filepath):
    #plt.imshow(logo_img)

    fig.savefig(filepath, format='jpg', dpi=200, bbox_inches='tight')
    return fig, plt


def plot_filter_logos3(W, figsize=(10,7), height=25, nt_width=10, norm=0, alphabet='dna', norm_factor=3, sort=True, filepath="kernel.jpg"):
    # 12x1x6x8
    W = np.squeeze(W.transpose([3, 2, 0, 1]))
    # 8x6x12x1
    num_filters = W.shape[0]
    num_rows = int(np.ceil(np.sqrt(num_filters)))
    grid = mpl.gridspec.GridSpec(num_rows, num_rows)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)

    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize);
    MAX_seq = np.max(np.abs(W[:,:4,:]))
    MAX_str = np.max(np.abs(W[:,4:,:]))
    ids = range(num_filters)
    b=[]
    b1=[]
    #c=[]
    if sort:
        for i in range(num_filters):
            b.append(-np.linalg.norm(W[i], ord=2))
            b1.append(np.linalg.norm(W[i], ord=1))
        ids = np.array(b).argsort()
        #print(b[ids])
    j =0
    for i in ids:
        plt.subplot(grid[j]);
        if norm_factor:
            W_seq = normalize_pwm(W[i,:4,:], factor=norm_factor, MAX=MAX_seq)
            W_str = normalize_pwm(W[i,4:,:], factor=norm_factor, MAX=MAX_str)
        else:
            W_seq = W[i,:4,:]
            W_str = W[i,4:,:]
        logo_seq = seq_logo(W_seq, height=height, nt_width=nt_width, norm=norm, alphabet=alphabet)
        logo_str = seq_logo_reverse(W_str, height=height, nt_width=nt_width, norm=norm, alphabet='pu', colormap='bw')
        spacer = np.zeros([1, W_str.shape[-1]*nt_width, 3], dtype=np.uint8)
        #spacer.fill(255)

        # build logo image
        logo_img = np.vstack([logo_seq, spacer,logo_str])

        title = 'L2: %.3f L1: %.3f' % (-b[i], b1[i])
        plt.title(title, fontsize='large')
        plt.yticks([])
        plt.xticks([])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

        plt.imshow(logo_img, interpolation='none')
        #plot_seq_logo(logo, nt_width=nt_width, step_multiple=None)


        #if np.mod(i, num_rows) != 0:

        j+=1
    #save_logo(logo_img, filepath):
    #plt.imshow(logo_img)

    fig.savefig(filepath, format='jpg', dpi=200, bbox_inches='tight')
    return fig, plt

def plot_seq_logo(logo, nt_width=None, step_multiple=None):
    plt.imshow(logo, interpolation='none')
    if nt_width:
        num_nt = logo.shape[1]/nt_width
        if step_multiple:
            step_size = int(num_nt/(step_multiple+1))
            nt_range = range(step_size, step_size*step_multiple)
            plt.xticks([step_size*nt_width, step_size*2*nt_width, step_size*3*nt_width, step_size*4*nt_width],
                        [str(step_size), str(step_size*2), str(step_size*3), str(step_size*4)])
        else:
            plt.xticks([])
        #plt.yticks([0, 50], ['2.0','0.0'])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
    else:
        plt.imshow(logo, interpolation='none')
        plt.axis('off');
    return plt

def show_struct_line_ax(ax, width, pwm_str):
    #print(pwm_str)
    height = 400
    base = 1720
    i=0
    x = np.zeros(width)
    for v in range(pwm_str.shape[0]):
        x[i*100:(i+1)*100] = (1-pwm_str[v])*height + base
        i+=1
    #x = np.linspace(0, 400, width) + 1300
    ax.plot(x, '-', color="r") #linewidth=20,
    #ax.set_ylim(base, base+ height)


def show_struct_line(ax, width, pwm_str):
    #ax2 = fig.add_axes([0.01,0.3,1,1])
    #ax2 = ax.twinx()
    #print(pwm_str)
    height = 400
    #base = 1720
    base = 0#304
    i=0
    x = np.zeros(width)
    for v in range(pwm_str.shape[0]):
        x[i*100:(i+1)*100] = pwm_str[v]*height + base
        #x[i*100:(i+1)*100] = (1-pwm_str[v])*height + base
        i+=1
    #x = np.linspace(0, 400, width) + 1300
    ax.plot(x, '-', color="r") #linewidth=20,
    ax.set_ylim(0, 400)
    #ax2.set_position([0,10,10,4])

def show_logo(logo_img):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 5))
    #ax = plt.subplot(1, 1, 1)
    plt.axis('off');
    plt.imshow(logo_img)
    plt.show()

def save_logo(logo_img, filepath):
    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 5))
    #ax = plt.subplot(1, 1, 1)
    plt.axis('off');
    plt.imshow(logo_img)


    fig.savefig(filepath, format='jpg', dpi=200, bbox_inches='tight')

def show_logo2(logo_img, pwm_str):
    import matplotlib.pyplot as plt
    #fig = plt.figure(figsize=(20, 5))
    _, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.imshow(logo_img)
    ax0.axis('off');
    ax0.set_position([0,0,2,1])
    ax1.axis('off');
    ax1.set_position([-0.1,0.36,2.2,0.12])
    width = logo_img.shape[1]
    show_struct_line(ax1, width, pwm_str)
    #plt.axis('off');
    plt.show()


def save_logo2(logo_img, pwm_str, filepath):
    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.imshow(logo_img)
    ax0.axis('off');
    ax0.set_position([0,0,2,1])
    ax1.axis('off');
    ax1.set_position([-0.1,0.36,2.2,0.12])
    width = logo_img.shape[1]
    show_struct_line(ax1, width, pwm_str)

    fig.savefig(filepath, format='jpg', dpi=200, bbox_inches='tight')

def get_nt_height(pwm, height, norm):
    alphabet = 'seq'

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq));
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            # if pwm[:, i] == -1:
            #     total_height = height
            #     print("-1")
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
        if alphabet == 'pu':
            heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
        else:
            heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

    return heights.astype(int)



def plot_seq_saliency_old(X, W, nt_width=100, norm_factor=3, show_plot=True, outdir="results"):


    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0)!=0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq = X[:4, plot_index]
    pwm_seq_logo = seq_logo(pwm_seq, height=nt_width, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')



    # sequence saliency logo
    seq_saliency = W[:4, plot_index]

    # get the heights of each nucleotide
    #heights = get_nt_height(pwm, height, norm)
    #remaining_height = np.sum(heights[:,i]);

    pwm_seq_saliency = normalize_pwm(seq_saliency, factor=norm_factor)
    pwm_seq_saliency_logo = seq_logo(pwm_seq_saliency, height=nt_width*5, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')


    # black line
    line1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    # spacer1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)
    # spacer1.fill(255)

    # spacing between seq and struct logo
    spacer2 = np.zeros([20, num_nt*nt_width, 3], dtype=np.uint8)
    spacer2.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt*nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1, spacer2, pwm_seq_logo, spacer2,
                          ])

    if show_plot:
        show_logo(logo_img)
    else:

        #filepath = outdir+"_seq.jpg"
        save_logo(logo_img,outdir)

    # return plot handles
    return logo_img

def plot_seq_saliency(X, W, nt_width=100, norm_factor=3, show_plot=True, outdir="results"):


    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0)!=0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq = X[:4, plot_index]
    pwm_seq_logo = seq_logo(pwm_seq, height=nt_width, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')



    # sequence saliency logo
    seq_saliency = W[:4, plot_index]

    # get the heights of each nucleotide
    #heights = get_nt_height(pwm, height, norm)
    #remaining_height = np.sum(heights[:,i]);

    pwm_seq_saliency = normalize_pwm(seq_saliency, factor=norm_factor)
    pwm_seq_saliency_logo = seq_logo(pwm_seq_saliency, height=nt_width*5, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')

    seq_sal = seq_saliency.T
    #print("str_sal:",str_sal.shape)
    seq_img = mat2img(seq_sal, (num_nt*nt_width,400))
    # black line
    line1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    # spacer1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)
    # spacer1.fill(255)

    # spacing between seq and struct logo
    spacer2 = np.zeros([20, num_nt*nt_width, 3], dtype=np.uint8)
    spacer2.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt*nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1,seq_img,line1, spacer2, pwm_seq_logo, spacer2,
                          ])

    if show_plot:
        show_logo(logo_img)
    else:

        #filepath = outdir+"_seq.jpg"
        save_logo(logo_img,outdir)

    # return plot handles
    return logo_img

def plot_seq_struct_saliency(X, W, nt_width=100, norm_factor=3, str_null=None, show_plot=True, outdir="results/"):


    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0)!=0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq = X[:4, plot_index]
    pwm_seq_logo = seq_logo(pwm_seq, height=nt_width, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')

    # structure logo
    pwm_struct = X[4:, plot_index]
    pwm_struct = normalize_pwm(pwm_struct, factor=norm_factor)
    if str_null is None:
        pwm_struct_logo = seq_logo_reverse(pwm_struct, height=nt_width*2, nt_width=nt_width, norm=0, alphabet='pu', colormap='bw')
    else:
        pwm_struct_logo = seq_logo_reverse_i(pwm_struct, height=nt_width*2, nt_width=nt_width, norm=0, alphabet='pu', colormap='bw', str_null=str_null)

    # sequence saliency logo
    seq_saliency = W[:4, plot_index]
    pwm_seq_saliency = normalize_pwm(seq_saliency, factor=norm_factor)
    pwm_seq_saliency_logo = seq_logo(pwm_seq_saliency, height=nt_width*5, nt_width=nt_width, norm=0, alphabet='rna', colormap='standard')

    # structure saliency logo
    struct_saliency = W[4:, plot_index]
    pwm_struct_saliency = normalize_pwm(struct_saliency, factor=norm_factor)
    pwm_struct_saliency_logo = seq_logo_reverse(pwm_struct_saliency, height=int(nt_width*8), nt_width=nt_width, norm=0, alphabet='pu', colormap='bw')


    seq_sal = seq_saliency.T
    #print("str_sal:",str_sal.shape)
    seq_img = mat2img(seq_sal, (num_nt*nt_width,400))

    str_sal = pwm_struct_saliency.T
    #print("str_sal:",str_sal.shape)
    #str_img = mat2img(str_sal[:,1:], (num_nt*nt_width,400))
    str_img = mat2img(str_sal.max(axis=1).reshape(-1,1), (num_nt*nt_width,400))
    #print("str_img:", str_img.shape)
    #print("width:", num_nt*nt_width)

    # black line
    line1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between seq and struct logo
    spacer2 = np.zeros([20, num_nt*nt_width, 3], dtype=np.uint8)
    spacer2.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt*nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1,seq_img,line1, spacer2, pwm_seq_logo, spacer2,line1,str_img,line1, spacer6,
                          pwm_struct_logo, line1, spacer6, pwm_struct_saliency_logo, ])
    #logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1,seq_img,line1, spacer2, pwm_seq_logo, spacer2,line1,str_img,line1 ])

    pwm_str = pwm_struct[1:,:].T
    # null icSHAPE value
    pwm_str[str_null.T==1] = -1
    if show_plot:
        show_logo2(logo_img, pwm_str)
    else:
        #filepath = outdir+"_pu.jpg"
        save_logo2(logo_img, pwm_str, outdir)
    return logo_img





def plot_pos_saliency(W, height=500, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):
    """
    pwm = normalize_pwm(W, factor=factor)
    pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    """
    # sequence saliency logo
    pwm = normalize_pwm(W, factor=norm_factor)
    logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # plot logo image
    plt.imshow(logo)
    plt.axis('off');

    # return figure and plot handles
    return plt



def plot_seq_pos_saliency(X, W, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):

    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0)!=0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq_logo = seq_logo(X[:,plot_index], height=nt_width, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # sequence saliency logo
    pwm_seq_saliency = normalize_pwm(W[:,plot_index], factor=norm_factor)
    pwm_seq_saliency_logo = seq_logo(pwm_seq_saliency, height=nt_width*5, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # black line
    line1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt*nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pwm_seq_saliency_logo, spacer6, line1, spacer1, pwm_seq_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off');

    # return plot handles
    return plt


def plot_neg_saliency(W, height=500, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):

    """
    num_rows = 2
    grid = mpl.gridspec.GridSpec(num_rows, 1)
    grid.update(wspace=0.2, hspace=0.00, left=0.1, right=0.2, bottom=0.0, top=0.05)

    fig = plt.figure(figsize=figsize);

    plt.subplot(grid[0])
    pwm = normalize_pwm(W, factor=factor)
    pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    if title:
        plt.title(title)

    plt.subplot(grid[1]);
    pwm = normalize_pwm(-W, factor=factor)
    neg_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(neg_logo[::-1,:,:], interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    fig.set_size_inches(150, 14)
    return fig, plt
    """
    num_nt = W.shape[1]

    # sequence logo
    pos_saliency = normalize_pwm(W, factor=norm_factor)
    pos_logo = seq_logo(pos_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # sequence saliency logo
    neg_saliency = normalize_pwm(-W, factor=norm_factor)
    neg_logo = seq_logo_reverse(neg_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # black line
    line1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([30, num_nt*nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pos_logo, spacer6, line1, spacer6, neg_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off');

    # return plot handles
    return plt


def plot_seq_neg_saliency(X, W, height=500, nt_width=100, alphabet='dna', norm_factor=3, colormap='standard'):
    """
    num_rows = 3
    grid = mpl.gridspec.GridSpec(num_rows, 1)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)

    fig = plt.figure(figsize=figsize);

    plt.subplot(grid[0])
    pwm = normalize_pwm(W, factor=factor)
    pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(pos_logo, interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    if title:
        plt.title(title)

    plt.subplot(grid[1])
    logo = seq_logo(np.squeeze(X), height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(logo, interpolation='none');
    plt.axis('off');

    plt.subplot(grid[2]);
    pwm = normalize_pwm(-W, factor=factor)
    neg_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
    plt.imshow(neg_logo[::-1,:,:], interpolation='none')
    plt.xticks([])
    plt.yticks([])
    #plt.yticks([0, 100], ['2.0','0.0'])
    #plt.yticks([0, 100], ['0.0','2.0'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    return fig, plt
    """
    # filter out zero-padding
    plot_index = np.where(np.sum(X, axis=0)!=0)[0]
    num_nt = len(plot_index)

    # sequence logo
    pwm_seq_logo = seq_logo(X[:,plot_index], height=int(height/5), nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)
    W = W[:,plot_index]

    pos_saliency = normalize_pwm(W, factor=norm_factor)
    pos_logo = seq_logo(pos_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # sequence saliency logo
    neg_saliency = normalize_pwm(-W, factor=norm_factor)
    neg_logo = seq_logo_reverse(neg_saliency, height=height, nt_width=nt_width, norm=0, alphabet=alphabet, colormap=colormap)

    # black line
    line1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)

    # space between seq logo and line
    spacer1 = np.zeros([10, num_nt*nt_width, 3], dtype=np.uint8)
    spacer1.fill(255)

    # spacing between seq and struct logo
    spacer2 = np.zeros([20, num_nt*nt_width, 3], dtype=np.uint8)
    spacer2.fill(255)

    # spacing between saliency logo and line
    spacer6 = np.zeros([60, num_nt*nt_width, 3], dtype=np.uint8)
    spacer6.fill(255)

    # build logo image
    logo_img = np.vstack([pos_logo, spacer6, line1, spacer2, pwm_seq_logo, spacer2, line1, spacer6, neg_logo])

    # plot logo image
    plt.imshow(logo_img)
    plt.axis('off');

    # return plot handles
    return plt



#------------------------------------------------------------------------------------------------
# helper functions

def fig_options(plt, options):
    if 'figsize' in options:
        fig = plt.gcf()
        fig.set_size_inches(options['figsize'][0], options['figsize'][1], forward=True)
    if 'ylim' in options:
        plt.ylim(options['ylim'][0],options['ylim'][1])
    if 'yticks' in options:
        plt.yticks(options['yticks'])
    if 'xticks' in options:
        plt.xticks(options['xticks'])
    if 'labelsize' in options:
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=options['labelsize'])
        ax.tick_params(axis='y', labelsize=options['labelsize'])
    if 'axis' in options:
        plt.axis(options['axis'])
    if 'xlabel' in options:
        plt.xlabel(options['xlabel'], fontsize=options['fontsize'])
    if 'ylabel' in options:
        plt.ylabel(options['ylabel'], fontsize=options['fontsize'])
    if 'linewidth' in options:
        plt.rc('axes', linewidth=options['linewidth'])


def subplot_grid(nrows, ncols):
    grid= mpl.gridspec.GridSpec(nrows, ncols)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)
    return grid


def load_alphabet(char_path, alphabet, colormap='standard'):

    def load_char(char_path, char, color):
        colors = {}
        colors['green'] = [10, 151, 21]
        colors['red'] = [204, 0, 0]
        colors['orange'] = [255, 153, 51]
        colors['blue'] = [0, 0, 204]
        colors['cyan'] = [153, 204, 255]
        colors['purple'] = [178, 102, 255]
        colors['grey'] = [160, 160, 160]
        colors['black'] = [0, 0, 0]

        img = mpimg.imread(os.path.join(char_path, char+'.eps'))
        img = np.mean(img, axis=2)
        x_index, y_index = np.where(img != 255)
        y = np.ones((img.shape[0], img.shape[1], 3))*255
        for i in range(3):
            y[x_index, y_index, i] = colors[color][i]
        return y.astype(np.uint8)


    colors = ['green', 'blue', 'orange', 'red']
    if alphabet == 'dna':
        letters = 'ACGT'
        if colormap == 'standard':
            colors = ['green', 'blue', 'orange', 'red']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    elif alphabet == 'rna':
        letters = 'ACGU'
        if colormap == 'standard':
            colors = ['green', 'blue', 'orange', 'red']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))


    elif alphabet == 'structure': # structural profile

        letters = 'PHIME'
        if colormap == 'standard':
            colors = ['blue', 'green', 'orange', 'red', 'cyan']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    elif alphabet == 'pu': # structural profile

        letters = 'PU'
        if colormap == 'standard':
            colors = ['cyan', 'purple']
        elif colormap == 'bw':
            colors = ['black', 'grey']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))
    elif alphabet == 'npu': # structural profile

        letters = 'IPU'
        if colormap == 'standard':
            colors = ['cyan', 'orange', 'purple']
        elif colormap == 'bw':
            colors = ['cyan', 'black', 'grey']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))
    elif alphabet == 'pui': # structural profile

        letters = 'PUI'
        if colormap == 'standard':
            colors = [ 'orange', 'purple', 'cyan']
        elif colormap == 'bw':
            colors = ['black', 'grey', 'cyan']
        chars = []
        for i, char in enumerate(letters):
            chars.append(load_char(char_path, char, colors[i]))

    return chars



def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):

    def get_nt_height(pwm, height, norm):

        def entropy(p):
            s = 0
            for i in range(len(p)):
                if p[i] > 0:
                    s -= p[i]*np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq));
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                # if pwm[:, i] == -1:
                #     total_height = height
                #     print("-1")
                total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
            if alphabet == 'pu':
                heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
            else:
                heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

        return heights.astype(int)


    # get the alphabet images of each nucleotide
    package_directory = os.path.dirname(os.path.abspath(__file__))
    char_path = os.path.join(package_directory,'chars')
    chars = load_alphabet(char_path, alphabet, colormap)

    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)
    #remaining_height = np.sum(heights[:,i]);

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)

    if alphabet == 'pu':
        max_height = height
    else:
        max_height = height*2
    #total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
    logo = np.ones((max_height, width, 3)).astype(int)*255;
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i]);
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i]);
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))

                # determine location of image
                height_range = range(remaining_height-nt_height[j], remaining_height)
                width_range = range(i*nt_width, i*nt_width+nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range+offset, width_range[m],k] = nt_img[:,m,k];

                remaining_height -= nt_height[j]

    return logo.astype(np.uint8)



def seq_logo_reverse(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):

    def get_nt_height(pwm, height, norm):

        def entropy(p):
            s = 0
            for i in range(len(p)):
                if p[i] > 0:
                    s -= p[i]*np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq));
        for i in range(num_seq):

            if norm == 1:
                total_height = height
            else:
                total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
            if alphabet == 'pu' :
                heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
            else:
                heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

        return heights.astype(int)


    # get the alphabet images of each nucleotide
    package_directory = os.path.dirname(os.path.abspath(__file__))
    char_path = os.path.join(package_directory,'chars')
    chars = load_alphabet(char_path, alphabet, colormap)
    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)

    if alphabet == 'pu':
        max_height = height
    else:
        max_height = height*2
    #total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
    logo = np.ones((max_height, width, 3)).astype(int)*255;
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = 0

        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))

                # determine location of image
                height_range = range(remaining_height, remaining_height+nt_height[j])
                width_range = range(i*nt_width, i*nt_width+nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range, width_range[m],k] = nt_img[:,m,k];
                remaining_height += nt_height[j]
    return logo.astype(np.uint8)


def seq_logo_reverse_i(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard', str_null=None):

    def get_nt_height(pwm, height, norm, str_null=None):

        def entropy(p):
            s = 0
            for i in range(len(p)):
                if p[i] > 0:
                    s -= p[i]*np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq));
        for i in range(num_seq):
            if str_null is not None and str_null[i]==1:
                continue
            if norm == 1:
                total_height = height
            else:
                total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
            if alphabet == 'pu' or alphabet == 'pui':
                heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
            else:
                heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

        return heights.astype(int)


    # get the alphabet images of each nucleotide
    package_directory = os.path.dirname(os.path.abspath(__file__))
    char_path = os.path.join(package_directory,'chars')
    if str_null is not None:
        alphabet = 'pui'
    chars = load_alphabet(char_path, alphabet, colormap)
    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm,str_null)
    max_height_pwm = heights.max()

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)

    if alphabet == 'pu' or alphabet == 'pui':
        max_height = height
    else:
        max_height = height*2
    #total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
    logo = np.ones((max_height, width, 3)).astype(int)*255;
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = 0

        if str_null is not None and str_null[i]==1:
            #print("I")
            nt_img = imresize(chars[2], (max_height_pwm, nt_width))
            height_range = range(remaining_height, remaining_height+max_height_pwm)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range, width_range[m],k] = nt_img[:,m,k];
            continue
        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))

                # determine location of image
                height_range = range(remaining_height, remaining_height+nt_height[j])
                width_range = range(i*nt_width, i*nt_width+nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range, width_range[m],k] = nt_img[:,m,k];

                remaining_height += nt_height[j]
    return logo.astype(np.uint8)
