from __future__ import print_function
import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([

    #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomAffine(degrees=360, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    #transforms.RandomResizedCrop(48, ratio=(0.8, 1.25)),

    transforms.RandomResizedCrop(224, scale=(0.8, 1.25), ratio=(0.8, 1.25)),
    #transforms.RandomResizedCrop(48),
    #transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    #brightness (float) – How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    #contrast (float) – How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    #saturation (float) – How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    #hue (float) – How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]
    transforms.ColorJitter(0.3, 0.4, 0.3, 0.3),

    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])


val_transforms = transforms.Compose([
    transforms.Scale((224, 224)),
    #transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    #transforms.RandomResizedCrop(48, scale=(0.9, 1), ratio=(0.8, 1.25)),
    #transforms.RandomResizedCrop(48),
    #transforms.ColorJitter(0.1, 0., 0.),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])
