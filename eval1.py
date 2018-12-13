from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from datetime import datetime as dt
import numpy as np
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='Galaxy ZOO')
parser.add_argument('--name', type=str, default='resnet50.csv')
parser.add_argument('--load', type=str)
parser.add_argument('--optimized', action="store_true", default=True)

args = parser.parse_args()
print(args)

### Data Initialization and Loading
#from data import data_transforms, val_transforms # data.py in the same folder
from galaxy import GalaxyZooDataset
from torch.utils.data import DataLoader

val_transforms = transforms.Compose([
                                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                                # transforms.RandomResizedCrop(args.input_size),
                                transforms.Resize(336),
                                # transforms.RandomHorizontalFlip(),
                                transforms.TenCrop((224,224)),
                                torchvision.transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                                #transforms.ToTensor(),
                                #transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                                ])


val_data = GalaxyZooDataset(train=False, transform=val_transforms)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False,
                                  num_workers=4, pin_memory=True, collate_fn=val_data.collate)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model_dnn import Net
from paper_2stn import Net
from nets import resnet
from nets import vgg
from nets import alexnet

if 'resnet50' in args.name:
    model = resnet.resnet50(optimized=args.optimized)
elif 'resnet101' in args.name:
    model = resnet.resnet101()
elif 'resnet18' in args.name:
    model = resnet.resnet18()
elif 'resnet34' in args.name:
    model = resnet.resnet34()
elif 'vgg16_bn' in args.name:
    model = vgg.vgg16_bn()
elif 'alex' in args.name:
    model = alexnet()
else:
    model = resnet.resnet152()

device = torch.device('cuda:0')

if args.load:
    model.load_state_dict(torch.load(args.load))
    print("Load sucessfully !", args.load)

model.to(device)
output_file = open('./results/' + args.name, "w")
head = 'GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n'
output_file.write(head)


def validation():
    model.eval()
    for meta in tqdm(val_loader):
        data = meta['image'].to(device)
        shape = data.shape
        data = data.reshape(-1, shape[2], shape[3], shape[4])
        names = meta['name']
        data = Variable(data, volatile=True)

        output = model(data).reshape(shape[0], 10, -1)
        #print(output.shape)
        output = output.mean(dim=1)
        #print(output.shape)


        for i in range(len(names)):
            name = names[i]
            strs = "{}".format(int(name))
            for j in range(37):
                strs = strs + ',{}'.format(float(output[i][j]))
            #print(strs)
            output_file.write(strs + '\n')

    print(dt.now(), 'Done. ')


print(dt.now(),'Start.') 
validation()
output_file.close()
