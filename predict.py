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
from PIL import Image
import json
import argparse
import function as f


# Load model with checkpoint
model = f.load_checkpoint('checkpoint.pth')

# Let the user chose the image file, epochs, learnrate
parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--file_path",
                    help="pass in your data directory here",
                    default='/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg',
                    required=False)
parser.add_argument("-epochs", "--epochs", type=int,
                    help="number of training iteration",
                    default=2, required=False)
parser.add_argument("-lr", "--learn_rate", type=float,
                    help="the learning rate",
                    default=0.0001, required=False)
parser.add_argument("-topk", "--topks", type=int,
                    help="number of top match results",
                    default=5, required=False)

args = parser.parse_args()
test = args.file_path
epochs = args.epochs
lr = args.learn_rate
topk = args.topks

# Load the label mapping
with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as c:
    cat_to_name = json.load(c)

# Return the classification of the input and its probability
result = (f.predict(test, model, cat_to_name, topk))

# Print the result
print(np.reshape(result[1:], (2, len(result[1]))))
print("\nThe top match flower category is " +
      result[1][0] + " with a probability of " + result[2][0])
