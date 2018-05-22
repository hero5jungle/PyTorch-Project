# Imports here
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import argparse
import function as f

# Parse in the image dataset path
parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--data_dir",
                    help="pass in your data directory here",
                    default='/home/workspace/aipnd-project/flowers')
parser.add_argument("-epochs", "--epochs", type=int,
                    help="number of training iteration",
                    default=3, required=False)
parser.add_argument("-lr", "--learn_rate", type=float,
                    help="the learning rate",
                    default=0.0001, required=False)
args = parser.parse_args()
data_dir = args.data_dir
epochs = args.epochs
learn_rate = args.learn_rate

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])

# TODO: Load the datasets with ImageFolder

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=40, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=40)
testloader = torch.utils.data.DataLoader(test_data, batch_size=40)


# TODO: Build and train your network
model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu1', nn.ReLU(inplace=True)),
    ('dropout1', nn.Dropout()),
    ('fc2', nn.Linear(4096, 1000)),
    ('relu2', nn.ReLU(inplace=True)),
    ('dropout2', nn.Dropout()),
    ('fc3', nn.Linear(1000, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

# Set parameters for training model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

# Train a model with a pre-trained network and compute the accuracy
model.train()
f.train(epochs, trainloader, valloader, model, criterion, optimizer)

# Validate the model
model.eval()
print("\nValidating the trained model on a different dataset. The accuracy result is below:")
f.validation(testloader, model, criterion)

# Save the model
checkpoint = {'filepath': data_dir,
              'model': models.vgg16(pretrained=True),
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'criterion': criterion,
              'epochs': epochs}
torch.save(checkpoint, 'checkpoint.pth')