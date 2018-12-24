from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import resnet
import datetime as dt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--net', type=str, default='resnet18', metavar='NET',
                    help="Name of the network module")
parser.add_argument('--name', type=str, default='smallbatch', metavar='N',
                    help="Name of the module")
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--dp', type=float, default=0.5, metavar='DP',
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--step', type=int, default=100, metavar='STEP',
                    help='steo (default: 100)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--load', type=str)
parser.add_argument('--pretrained', action='store_true')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
### Data Initialization and Loading
from data import initialize_data, data_transforms, val_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=val_transforms),
    batch_size=args.batch_size//2, shuffle=False, num_workers=4)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
#model = Net(args.dp)
if 'resnet50' in args.net:
    model = resnet.resnet50(args.pretrained, dp = args.dp)
    #model = resnet.resnet50(True, dp = args.dp)
if 'resnet18' in args.net:
    model = resnet.resnet18(args.pretrained, dp = args.dp)

if args.load:
    model.load_state_dict(torch.load(args.load))
#model.load_state_dict(torch.load(['model_latest.pth'])['state_dict'])

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step)
device = torch.device('cuda:0')
model.to(device)
best_accu = 0

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() 
        #output = model(data) 
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        #scheduler.step()
        if batch_idx % args.log_interval == 0:
            print(dt.datetime.now(), 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], correct, args.log_interval * len(data), 100. * int(correct )/ (args.log_interval * len(data) )))
            correct = 0;

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()


    validation_loss /= len(val_loader.dataset)
    print(dt.datetime.now(), '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. *int( correct) / len(val_loader.dataset)))

    return 100. * int(correct) / len(val_loader.dataset)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    accu = validation()
    scheduler.step()
    model_file = 'results/' + args.name +'/model_' + str(epoch) +'_{:.2f}'.format(accu) + '.pth'
    if accu > best_accu and accu > 96:
        best_accu = accu
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
