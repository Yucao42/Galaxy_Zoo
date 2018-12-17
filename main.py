from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime as dt
import numpy as np
from nets.resnet import partial_nll
from custom import OptimisedDivGalaxyOutputLayer 

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--name', type=str, default='resnet50', metavar='NM',
                    help="name of the training")
parser.add_argument('--load', type=str,
                    help="load previous model to finetune")
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--no_dp', action='store_true', default=False,
                    help="if there is no dropout")
parser.add_argument('--lock_bn', action='store_true', default=False,
                    help="if there is no BN gradient")
parser.add_argument('--sigmoid', action='store_true', default=False,
                    help="if there is sigmoid")
parser.add_argument('--optimized', action='store_true', default=False,
                    help="if there optimized normalization")
parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--step', type=int, default=10, metavar='S', 
                    help='lr decay step (default: 5)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--p', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, metavar='WD',
                    help='Weight decay (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import data_transforms, val_transforms # data.py in the same folder
from galaxy import GalaxyZooDataset
from torch.utils.data import DataLoader

train_data = GalaxyZooDataset(train=True, transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, collate_fn=train_data.collate)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model_dnn import Net
from paper_2stn import Net
from nets import resnet
from nets import vgg
from nets import alexnet

if 'resnet50' in args.name:
    model = resnet.resnet50(True, lock_bn=args.lock_bn, sigmoid=args.sigmoid, optimized=args.optimized)
elif 'resnet101' in args.name:
    model = resnet.resnet101(True)
elif 'resnet18' in args.name:
    model = resnet.resnet18(True, lock_bn=args.lock_bn, sigmoid=args.sigmoid)
elif 'resnet34' in args.name:
    model = resnet.resnet34(True, lock_bn=args.lock_bn, sigmoid=args.sigmoid)
elif 'vgg16_bn' in args.name:
    model = vgg.vgg16_bn(True)
elif 'alex' in args.name:
    model = alexnet(True)
else:
    model = resnet.resnet152(True)

device = torch.device('cuda:0')

if args.load:
    model.load_state_dict(torch.load(args.load))
    try:    
        model.load_state_dict(torch.load(args.load))
        print("Load sucessfully !", args.load)
    except:
        print("Training from scratch!")

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step)
least_mse = np.inf

normalizer = OptimisedDivGalaxyOutputLayer() 
kl_func = nn.KLDivLoss()
epi = 1e-9

use_kl = True
dual_custom = True
focal_mse = True

# Scale factor to the first question
sf = 1
def weighted_mse(score, target, wts):
    mse = (score - target) ** 2
    #mse_loss = torch.mm(mse, wts).mean()
    wts = wts.cuda()
    mse_loss = (mse * wts).mean()
    wts_step = mse.mean(dim=0)
    wts_step = wts_step / wts_step.sum()
    return mse_loss, wts_step.reshape(1, -1) 

def train(epoch):
    model.train()
    correct = 0
    loss_total = 0
    loss_1 = 0
    loss_2 = 0
    loss_step  = 0
    wts = torch.ones(37)/ 37
    wts.to(device)
    wts = Variable(wts, requires_grad=False)
    wts_step = torch.ones(37).reshape(1,37)/ 37
    wts_step = wts_step.cuda()
    for batch_idx, meta in enumerate(train_loader):
        data, target = meta['image'].to(device), meta['prob'].to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if focal_mse:
            loss, wts_batch = weighted_mse(output, target, wts) 
            wts_step = torch.cat([wts_step, wts_batch],dim=0) 
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            loss_step  += float(loss.item())
        elif dual_custom:
            cls_res = normalizer.answer_probabilities(output)
            cls_gts = normalizer.answer_probabilities(target)
            #loss_1 = 0.01 * kl_func(cls_res.log(), cls_gts)
            loss_1 = F.mse_loss(cls_gts, cls_res)
            loss_2 =  F.mse_loss(output, target)
            loss = loss_1 + 3 * loss_2
            #loss = loss_1 + 2 * loss_2
            #loss = F.mse_loss(output, target) + F.kl_div(output[:,:3].float(), target[:,:3])
            loss.backward()
            optimizer.step()
            loss_total += float(loss_2.item())
            loss_step  += float(loss.item()) 
        elif  use_kl:
            cls_res = output[:, :3] + 1e-10
            cls_gts = target[:, :3] + 1e-10
            #loss_1 = sf * F.mse_loss(cls_gts, cls_res)
            loss_1 = 0.01 * kl_func(cls_res.log(), cls_gts)
            loss_2 =  F.mse_loss(output, target)
            loss = loss_1 + loss_2
            #loss = F.mse_loss(output, target) + F.kl_div(output[:,:3].float(), target[:,:3])
            loss.backward()
            optimizer.step()
            loss_total += float(loss_2.item())
            loss_step  += float(loss.item())
        else:
            #loss =  kl_func((output+epi).log(), target+epi)

            loss =  F.l1_loss(output, target)
            #loss = F.mse_loss(output, target) + F.kl_div(output[:,:3].float(), target[:,:3])
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            loss_step  += float(loss.item())

        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            print(dt.now(), 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.sqrt(loss_step / (args.log_interval ) )))
            loss_step = 0
            if focal_mse and batch_idx % 900 == 0:
                wts = Variable(wts_step.mean(dim=0).reshape(-1), requires_grad=False)
                wts_step = wts.reshape(1, -1)
                wts_step, wts = wts_step.cuda(), wts.cuda()
                print('Weights ', wts_step)

    print("\nTraining MSE loss: ", np.sqrt(loss_total * 1.0 / len(train_loader)))
    return loss_total * 1.0 / len(train_loader)       

for epoch in range(1, args.epochs + 1):
    loss = train(epoch)
    scheduler.step()
    model_file = "models/resnet/" + args.name +'/model_best.pth'
    #model_file = "models/resnet/" + args.name +'/model_' + str(epoch) +'_{:.9f}'.format(loss) + '.pth'
    if loss < least_mse :
        least_mse = loss
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file )
