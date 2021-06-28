#!/usr/bin/env python
import os, sys, pdb, h5py
import os.path
import errno
import numpy as np
import random
import torch
import pandas as pd
import torch.utils.data as data

    
class IcShape(data.Dataset):
    # data_path1="data/pdb_perchain_npz_4_rer/gt_name_1_sp"
    def __init__(self, data_path, name, args=None, ss_type='seq', is_test=False):
        
        if args is not None and args.generate_data:
            print("regression: "+str(args.regression)+", use_label: "+str(args.use_label)+", shuffle_neg:"+str(args.shuffle_neg))
            df = pd.read_csv(os.path.join(data_path, name+'.tsv'), sep='\t')
            #rnac_set = df['Type'].as_matrix()
            #name      = df['name'].as_matrix()
            sequences = df['Seq'].as_matrix()
            icshapes   = df['icshape'].as_matrix()
            targets   = df['Score'].as_matrix()
            if args.regression :
                # norm to [0,1]
                targets = nn.functional.tanh(targets)
            else:
                if args.use_label:
                    targets   = df['label'].as_matrix().reshape(-1,1)
                else:
                    # sequences = sequences[targets>=1] 
                    # icshapes  = icshapes[targets>=1] 
                    targets[targets>=1] = 1
                    # targets = targets[targets>=1].reshape(-1,1)
                    targets[targets<1] = 0
            
            index = torch.randperm(targets.shape[0])
            sequences = sequences[index] 
            icshapes  = icshapes[index] 
            targets   = targets[index].reshape(-1,1)
                

            self.dataset = encoding_seq(sequences, icshapes, targets, ss_type, shuffle_neg=args.shuffle_neg)
            print("targets:",len(targets), np.unique(targets,return_counts=True))
            print("dataset size:", len(self.dataset))
            np.save(os.path.join(data_path, name), self.dataset)
            print("npz file saved:", os.path.join(data_path, name)+".npy")
        else:
            filename = os.path.join(data_path, name+'.npy')
            print("Loading data:",filename)
            self.dataset = np.load(filename,allow_pickle=True)
            
            
        # train, valid, test = load_dataset_hdf5(data_path, ss_type=ss_type)
        # train, valid, test = process_data(train, valid, test, method=normalize_method)
        test_datasize = len(self.dataset) // 3 
        if is_test:
            self.dataset = self.dataset[:test_datasize]
        else:
            self.dataset = self.dataset[test_datasize:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y


    def __len__(self):
        return len(self.dataset)

def encoding_seq(sequences, icshapes, targets, ss_type, max_length=None, shuffle_neg=False):
    """convert DNA/RNA sequences to a one-hot representation"""
    zero = np.array([0]).reshape(-1,1)
    one_hot_seq = []
    i = 0
    for seq in sequences:
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
        #print(one_hot.shape)

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])
        if ss_type =='pu':
            # icshape
            icshape = icshapes[i].split(',')
            ti = [float(t) for t in icshape]
            ti = np.array(ti).reshape(1,-1)
            print("ti:",ti.shape)
            encoding = np.concatenate([one_hot, ti], axis=0).T
        else:
            encoding = one_hot.T


        one_hot_seq.append((encoding, targets[i]))
        if shuffle_neg:
            encoding_perm = encoding[torch.randperm(encoding.shape[0])]
            one_hot_seq.append((encoding_perm, zero))
            # encoding_perm = encoding[torch.randperm(encoding.shape[0])]
            # one_hot_seq.append((encoding_perm, zero))
        
        i += 1
    return tuple(one_hot_seq)


class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order

        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.from_numpy(x[0]).float()  for x in sorted_batch]
        # print(sequences[0].shape)
        
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.FloatTensor([len(x)for x in sequences])
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.FloatTensor([torch.from_numpy(x[1]) for x in sorted_batch])
        return sequences_padded, lengths, labels

if __name__ == "__main__":
    import torch 
    
    
    data_path = 'data'
    name='cds'
    ss_type = 'pu'
    from net_ics import RNANet
    model = RNANet(input_dim=5,hidden_dim=20,output_dim=2,num_layers=1,biFlag=True)
    kwargs = {'num_workers': 2, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(IcShape(data_path,name, ss_type), \
        batch_size=8, shuffle=True, collate_fn=PadSequence(), **kwargs)
    for batch_idx, batch in enumerate(train_loader):
        model(batch)