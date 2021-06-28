from __future__ import print_function
import argparse, os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import warnings
warnings.filterwarnings("ignore")

from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np

from var.loader import IcShape, PadSequence
import var.net_ics as arch

from utils.xprint import log_print
from utils.checkpoint import best_checkpoint, make_directory
from utils.utils import acc_auc, clip_gradient_model, GradualWarmupScheduler


global writer, minloss, best_auc, best_epoch
minloss = 999
best_auc = 0
best_epoch = 0
early_stop_epoch = 40

    
def train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion):
    # scheduler.step()
    # lr = scheduler.get_lr()[0]
    lr = optimizer.param_groups[0]['lr']
    model.train()
    NUM_BATCH = len(train_loader.dataset)//args.batch_size
    acc_lst = []
    auc_lst = []
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        output, y = model(batch)
        y = y.view(-1,1)
        loss = criterion(output, y)
        cls_loss_, mask_loss_ = 0, 0
        acc_, auc_ = acc_auc(output, y)
        total_loss += loss.item()
        acc_lst.append(acc_)
        auc_lst.append(auc_)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            line = '{} \t Train Epoch: {} {:2.0f} Loss: {:.4f} '.format(\
            args.p_name, epoch, 100.0 * batch_idx / len(train_loader),  loss.item())
            print(line)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
    total_loss /= NUM_BATCH
    acc = 100. * np.mean(acc_lst)
    auc = np.mean(auc_lst)

    line='{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.4f}'.format(\
         args.p_name, epoch, total_loss, acc, auc, lr)#scheduler.get_lr()[0])
    log_print(line, color='green', attrs=['bold'])

    if args.tfboard:
        writer.add_scalar('loss/train', total_loss, epoch)
        writer.add_scalar('acc/train', acc, epoch)
        writer.add_scalar('AUC/train', auc, epoch)

def test(args, model, device, test_loader, epoch, model_path, criterion):
    global minloss, best_auc, best_epoch
    model.eval()
    test_loss = 0
    correct = 0
    NUM_BATCH = len(test_loader.dataset)//args.batch_size
    acc_lst = []
    auc_lst = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            output, y = model(batch)
            y = y.view(-1,1)
            loss = criterion(output, y)
            # cls_loss_, mask_loss_ = 0, 0
            acc_, auc_ = acc_auc(output, y)
            test_loss += loss.item() # sum up batch loss
            acc_lst.append(acc_)
            auc_lst.append(auc_)
    test_loss /= NUM_BATCH
    acc = 100. * np.mean(acc_lst)
    auc = np.mean(auc_lst)

    color='green'
    if not args.test and test_loss < minloss:
        minloss = test_loss
        best_auc = auc
        best_epoch = epoch
        color='red'
        filename = "{}/{}".format(model_path, args.arch+".pkl", auc)
        torch.save(model.state_dict(), filename)
    line='{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f})'.format(\
         args.p_name, epoch, test_loss, acc, auc, best_auc)
    log_print(line, color=color, attrs=['bold'])
    if args.tfboard:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/test', acc, epoch)
        writer.add_scalar('AUC/test', auc, epoch)
    return test_loss
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

def main():
    global writer, best_epoch
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch RBP Example')
    parser.add_argument('--datadir', required=True, help='data name')
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
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
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
    parser.add_argument('--exp_name', type=str, default="cnn", metavar='N',
                        help='experiment name')
    parser.add_argument('--p_name', type=str, default="cds", metavar='N',
                        help='protein name')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--restore_best', action='store_true', help='restore_best')
    parser.add_argument('--out_dir', type=str, default="motif_dir", help='output directory')
    parser.add_argument('--hidden_dim', type=int, default=20, help='number of epochs to train for')
    #parser.add_argument('--motif_w', action='store_true', help='motif weight')
    parser.add_argument('--nstr', type=int, default=1, help='number of vector encoding for structure data')
    parser.add_argument('--ss_type', type=str, default="pu", help='output directory')
    parser.add_argument('--tfboard', action='store_true', help='tf board')
    parser.add_argument('--generate_data', action='store_true', help='generate data')
    parser.add_argument('--shuffle_neg', action='store_true', help='generate data')
    parser.add_argument('--regression', action='store_true', help='generate data')
    parser.add_argument('--use_label', action='store_true', help='generate data')
    # parser.add_argument('--shuffle_neg', action='store_true', help='generate data')
    

    #parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
    #                    help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.001,
                        help='scale sparse rate (default: 0.0001)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    if args.ss_type == 'pu':
        args.nstr = 1
        ss_type = 'pu'
    else:
        args.nstr = 0
        ss_type = 'seq'

    if args.tfboard:
        writer = SummaryWriter("runs/"+args.exp_name)

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    #data_path  = args.datadir + "/" + args.p_name + ".h5"
    model_dir = "models" + "/" + args.exp_name 
    make_directory(model_dir, "_"+ss_type+"_1")
    model_path = model_dir +"/_"+ss_type+"_1"
    # if os.path.exists(model_path):
    #     os.path.makedirs(model_path)
    train_loader = torch.utils.data.DataLoader(IcShape(args.datadir, args.p_name, args, ss_type), \
        batch_size=args.batch_size, shuffle=True, collate_fn=PadSequence(), **kwargs)
    if args.generate_data:
        sys.exit(0)
    test_loader  = torch.utils.data.DataLoader(IcShape(args.datadir, args.p_name, args, ss_type, is_test=True), \
        batch_size=8, shuffle=False, collate_fn=PadSequence(), **kwargs)
    print("Train set:", len(train_loader.dataset))
    print("Test  set:", len(test_loader.dataset))


    print("Network Arch:", args.arch)
    print("Data Type:   ", ss_type)
    model = getattr(arch, args.arch)(input_dim=args.nstr+4, hidden_dim=args.hidden_dim, output_dim=1, num_layers=2, biFlag=True) 
    if args.restore_best:
        best_ckpt_file = model_path+"/"+args.arch+".pkl"
        print("loading best model:", best_ckpt_file)
        param = torch.load(best_ckpt_file)
        model.load_state_dict(param)
    if args.ngpu>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model.cuda()
    arch.param_num(model)



    #model = Net().to(device)
    if args.optimizer=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    #      optimizer, float(args.niter), eta_min=args.learning_rate_min)
    scheduler = GradualWarmupScheduler(
          optimizer, multiplier=2, total_epoch=float(args.niter), after_scheduler=None)

    if args.regression:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.BCELoss()

    test_loss = test(args, model, device, test_loader, 0, model_path, criterion)
    if args.test:
        sys.exit(0)

    for epoch in range(1, args.niter + 1):
        scheduler.step(epoch)
        train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion)
        test_loss = test(args, model, device, test_loader, epoch, model_path, criterion)
        if epoch - best_epoch > early_stop_epoch:
            print("Early stop at %d, %s "%(epoch, args.exp_name))
            break

    #if (args.save_model):
    #    torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()
