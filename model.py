#!/usr/bin/env python
# coding: utf-8

# # 1.Problem definition:
# I want to train a convolutional neural network, train a model to classify images from the CIFAR10 dataset into 10 categories.
# 
# This is second implementation of my model. First was on MNIST
# 

# In[ ]:


## to run this program, make sure that you have install: 
## anaconda or another python distribution 
## install pytorch torchvision cpuonly -c pytorch (if you dont have CUDA-capable GPU )
## install pytorch torchvision pytorch-cuda -c pytorch -c nvidia (if you have GPU with CUDA )
## install streamlit


# 2. Data collrcting: importing PyTorch, its neural network module, the torchvision library, NumPy, and the pyplot module from the matplotlib library.

# In[ ]:

import torch
import torch.nn as nn
import torchvision
import math
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
#device=torch.device("cpu:0")
#device=torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. Data preparation: defining a data transformation that converts the input images to PyTorch tensors and normalizes them, setting the batch size, and creating data loaders for the CIFAR10 dataset.

# In[ ]:


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4 # Cannot be too much when training on cpu

trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 4. Model Alhorithm: defining a ConvModel class that extends the PyTorch nn.Module class. This class defines the architecture of the CNN model, which consists of two convolutional layers followed by two fully connected layers. The forward() method implements the forward pass of the model.

# In[ ]:


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels= 32,
            kernel_size=3, 
            padding=1)
        
        self.pool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2)
        
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3,
            padding=1)
        
        self.fc1 = nn.Linear(
            in_features=64 * 8 * 8, 
            out_features=512)
        
        self.fc2 = nn.Linear(in_features=512, 
                             out_features=10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = ConvModel().to(device)
model


# 4.1 Model optimization: defining the loss function (cross-entropy) and the optimizer (stochastic gradient descent).

# In[ ]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 5. Model train: training the model for two epochs using the training data. In each epoch, you are iterating over the batches in the training data, computing the forward and backward passes, and updating the weights of the model using the optimizer.

# In[ ]:


model.train()
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


# 6. Evaluation: evaluating the performance of the trained model on the test data by computing the accuracy of the model. You are setting the model to evaluation mode to turn off dropout and batch normalization, and then iterating over the test data to compute the accuracy.

# In[ ]:


correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))


# This convolutional neural network (CNN) is a deep learning model that is trained on the CIFAR-10 dataset for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model architecture consists of two convolutional layers and two fully connected layers. The first convolutional layer has 32 filters of size 3x3, and the second convolutional layer has 64 filters of size 3x3. Both convolutional layers are followed by a max pooling layer that reduces the spatial dimensions of the feature maps. The output of the second max pooling layer is flattened and fed into two fully connected layers, with the first one having 512 neurons and the second one having 10 neurons (one for each class in the dataset). The ReLU activation function is used after each convolutional and fully connected layer, except for the last one, which uses a softmax activation function to output the probabilities of the input image belonging to each class. The model is trained using stochastic gradient descent (SGD) with a learning rate of 0.001 and a momentum of 0.9, and the cross-entropy loss function is used as the optimization objective.
# 

# 7. implementation: i used Streamlit library, and now i can create a interface that allows users to interact with your model. 

# In[ ]:


torch.save(model, 'cifar10_model.pth')

