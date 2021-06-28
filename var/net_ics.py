import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def param_num(model):
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("total parameters:     ", num_param0)
    #print("trainable parameters: ", num_param1)
    print("===========================")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0-num_param1)
    print("===========================")

class RNANet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biFlag, dropout=0.5):
        
        super(RNANet, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.device=torch.device("cuda")
        if(biFlag):
            self.bi_num=2
        else:
            self.bi_num=1
        self.biFlag=biFlag
        #self.gru = nn.GRU(5, 20, 2, batch_first=True)  # Note that "batch_first" is set to "True"
        self.bilstm=nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=biFlag)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
            nn.Sigmoid()
            # nn.LogSoftmax(dim=1)
        )
        # nn.LogSoftmax(dim=2)

    def forward(self, batch):
        x, x_lengths, y = batch
        return self.forward2(x, x_lengths, y)

    def forward2(self, x, x_lengths, y):
        x,y=x.to(self.device),y.to(self.device)
        x_pack = pack_padded_sequence(x, x_lengths, batch_first=True)
        self.bilstm.flatten_parameters()
        out_rnn, hidden = self.bilstm(x_pack)
        out_rnn, length = pad_packed_sequence(out_rnn, batch_first=True)
        
        # use mean
        last_tensor = out_rnn#[row_indices, :, :]
        last_tensor = torch.mean(last_tensor, dim=1)
        
        out=self.fc(last_tensor)
        return out, y

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=11, padding=5, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResRNANet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biFlag, dropout=0.5):
        
        super(ResRNANet, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.device=torch.device("cuda")
        if(biFlag):
            self.bi_num=2
        else:
            self.bi_num=1
        self.biFlag=biFlag
        self.conv1_out_n = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_out_n, kernel_size=(11,self.input_dim), padding=(5,0)),
            nn.BatchNorm2d(self.conv1_out_n),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(                   
            nn.Conv2d(self.conv1_out_n, self.conv1_out_n*4,   kernel_size=1, stride=1, bias=False),                    
            nn.BatchNorm2d(self.conv1_out_n*4)
        )
        self.resblock = ResidualBlock(in_channels=self.conv1_out_n, out_channels=self.conv1_out_n*4, 
            kernel_size=(11,1), padding=(5,0),downsample=self.downsample)
        self.input_dim = self.conv1_out_n*4
        self.bilstm=nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=biFlag)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
            nn.Sigmoid()
        )

    def forward(self, batch):
        x, x_lengths, y = batch
        return self.forward2(x, x_lengths, y)

    def forward2(self, x, x_lengths, y):
        x,y=x.to(self.device),y.to(self.device)
        length = x.shape[1]
        x = x.view(-1, 1, length, x.shape[2])
        x = self.conv1(x)  
        x = self.resblock(x)  
        x = x.view(-1, self.input_dim, length).transpose(1,2) 

        x_pack = pack_padded_sequence(x, x_lengths, batch_first=True)
        self.bilstm.flatten_parameters()
        out_rnn, hidden = self.bilstm(x_pack)

        out_rnn, length = pad_packed_sequence(out_rnn, batch_first=True)
        
        # use mean
        last_tensor = out_rnn #[row_indices, :, :]
        last_tensor = torch.mean(last_tensor, dim=1)
        
        out=self.fc(last_tensor)
        return out, y