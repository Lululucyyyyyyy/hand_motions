import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from cnn_utils import *
import glob
import random
from tensorflow.keras import layers
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pylab import imshow
import torchvision
import io
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn.init as init
#from onnx_tf.backend import prepare
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

#directions = ['five', 'one']
def load_images(globpath):
    for i, image in enumerate(globpath):
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = img.resize((224, 224))
            np_img = np.array(img)
            my_list.append(np_img)

my_list = []
load_images(glob.glob("dataset3/five*.JPG"))
load_images(glob.glob("dataset3/one*.JPG"))
my_list = np.array(my_list)
print('my_list.shape: ', my_list.shape)

# 0 five (66), 1 one (66)
labels_ = [0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1]
print("num labels: ",len(labels_))
labels_ = np.array(labels_)

p = np.random.permutation(len(my_list))
X_orig = my_list[p]
Y_orig = labels_[p]

X_test_orig = X_orig[0:15]
Y_test_orig = Y_orig[0:15]
X_dev_orig = X_orig[16:31]
Y_dev_orig = Y_orig[16:31]
X_train_orig = X_orig[32:len(labels_)]
Y_train_orig = Y_orig[32:len(labels_)]

X_train = X_train_orig/255
X_test = X_test_orig/255
X_dev = X_dev_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 2).T
Y_test = convert_to_one_hot(Y_test_orig, 2).T
Y_dev = convert_to_one_hot(Y_dev_orig, 2).T

X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
Y_train = torch.FloatTensor(Y_train)
X_dev = torch.FloatTensor(X_dev).permute(0, 3, 1, 2)
Y_dev = torch.FloatTensor(Y_dev)
X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
Y_test = torch.FloatTensor(Y_test)

def random_mini_batches(X, Y, mini_batch_size = 10, seed = 0):
    m = X.shape[0]                
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 11)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 256, 5)
        self.conv3 = nn.Conv2d(256, 384, 3)
        self.conv4 = nn.Conv2d(384, 384, 3)
        self.conv5 = nn.Conv2d(384, 256, 3)
        self.fc6 = nn.Linear(256*21*21, 120)
        self.fc7 = nn.Linear(120, 84)
        self.fc8 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        num = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train():
    print('==========Training==========')
    costs = []
    for epoch in range(5):
        running_loss = 0.0
        minibatches = random_mini_batches(X_train, Y_train)
        for i, minibatch in enumerate(minibatches, 0):
            #minibatch size = 10

            inputs, labels = minibatch

            optimizer.zero_grad()

            outputs = net(inputs)
            predicted = torch.argmax(outputs, 1)
            #print('real:      ', torch.argmax(labels, 1))
            #print('predicted: ',predicted)
            #print('outputs:    ',outputs)
            loss = criterion(outputs, torch.argmax(labels, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            temp = loss.item()
            costs.append(loss.item())
            #if i == 0 or i == 5:
            print('[%d, %5d] loss: %.3f' %  (epoch + 1, i + 1, running_loss))
            costs.append(running_loss)
            running_loss = 0.0
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Cost")
    plt.show()
    eval("train")

def dev():
    print('==========Validating==========')
    images = X_dev
    labels = Y_dev
    
    imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    my_labels = torch.argmax(labels, 1)
    print('GroundTruth: ', my_labels)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   ', predicted)
    eval("dev")

def test():
    print('==========Testing==========')
    images = X_test
    labels = Y_test
    
    imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    my_labels = torch.argmax(labels, 1)
    print('GroundTruth: ', my_labels)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   ', predicted)
    eval("test")

def eval(dataset):
    correct = 0
    total = 0
    if dataset == "train":
        images = X_train
        labels = Y_train
    elif dataset == "dev":
        images = X_dev
        labels = Y_dev
    elif dataset == "test":
        images = X_test
        labels = Y_test

    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        the_labels = torch.argmax(labels, 1)
        total += len(images)
        print('total:   ', total)
        correct += (predicted == the_labels).sum().item()
        print('correct: ', correct)
    print('Accuracy of the network on {}: %d %%'.format(dataset) % (100 * correct / total))

train()
dev()
test()
print('the end')

