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
from PIL import Image


def load_checkpoint(filepath='/home/workspace/paind-project/checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.eval()
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def train(epochs, trainloader, valloader, model, criterion, optimizer):
    model.train()
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        for images, labels in iter(trainloader):
            steps += 1
            inputs, targets = Variable(images), Variable(labels)
            # Check for GPU and place tensors accordingly
            cuda = torch.cuda.is_available()
            if cuda:
                model.cuda()
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            # Print out the results: running_loss, test_loss, accuracy

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                test_loss = 0
                for ii, (images, labels) in enumerate(valloader):
                    inputs = Variable(images, volatile=True)
                    labels = Variable(labels, volatile=True)
                    inputs, labels = inputs.cuda(), labels.cuda()
                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(valloader)))

                running_loss = 0
                model.train()

def validation(dataset, model, criterion):
    model.eval()
    accuracy = 0
    test_loss = 0
    for ii, (images, labels) in enumerate(dataset):
        inputs = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        # Check for GPU and place tensors accordingly
        cuda = torch.cuda.is_available()
        if cuda:
            model.cuda()
            inputs, labels = inputs.cuda(), labels.cuda()

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(dataset)),
          "Test Accuracy: {:.3f}".format(accuracy/len(dataset)))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    base = 256
    image = Image.open(image)
    w, h = image.size

    # Check which side is shorter and set that side = 256
    if w > h:
        w, h = base*(w/h), base
    elif w < h:
        w, h = base, base*(h/w)

    # Resize image
    image.thumbnail((w, h), Image.ANTIALIAS)

    # Crop image
    crop = image.crop((w//2 - 224//2, h//2 - 224//2,
                       w//2 + 224//2, h//2 + 224//2))

    # Normalize the image
    np_image = np.array(crop)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalized_image = (np_image - mean) / std
    final_image = normalized_image.transpose((2, 0, 1))

    return final_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when
    # displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, label_map, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    # Process the image, pass it into cuda or cpu torch tensor and wrap it in
    # variable
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        model_input = Variable(torch.from_numpy(
            process_image(image_path))).unsqueeze(0).float().cuda()

    # Pass the Variable through the model in eval mode
    model.eval()
    output = model.forward(model_input)
    # Convert logSoftmax of output to probability through expential
    ps = torch.exp(output)

    # Sort top 5 probability outputs
    topks_idx = ps.data.sort()[1][0][-topk:]
    classes = [label_map[str(idx+1)] for idx in topks_idx]
    topks_probs = ps.data.sort()[0][0][-topk:]
    topks_probs = ['%.3f' % elem for elem in topks_probs]

    return list(topks_idx), list(reversed(classes)), list(reversed(topks_probs))