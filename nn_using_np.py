from torch.nn import Module, Conv2d, Linear, AdaptiveAvgPool2d
from torch.nn.functional import relu
import numpy as np
import cv2
import os
import torch
import torch.nn as nn

data_dir = '/home/salahuddin/projects/Deeplearning_Practice/mix/'

def load_data(data_dir):
    X, y = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_dir, file_name))
            img = cv2.resize(img, (64, 64))
            img = torch.tensor(img).permute(2, 0, 1) / 255.0
            X.append(img)
            if 'cat' in file_name:
                y.append(1)
            else:
                y.append(0)
    X = torch.stack(X) # Convert individual components of a list into a full tensor
    y = torch.tensor(y)
    y = y.reshape(-1, 1)
    # print('test', X.shape)
    # y1 = torch.tensor(1)
    # y = torch.add(y, y1)

    return X, y

class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=4) 
        self.conv3 = Conv2d(in_channels=64, out_channels=256, kernel_size=5) 
        self.avgPooling = AdaptiveAvgPool2d(output_size=(256, 256))
        self.fc1 = Linear(in_features=256, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=64)
        self.out = Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgPooling(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.avgPooling(x)
        x = self.conv3(x)
        x = relu(x)

        x = self.avgPooling(x)    
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.out(x)
        return x

model = Network()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_loader = load_data(data_dir=data_dir)
N_train = len(train_loader[0])
n_epochs = 3

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        # print(train_loader[0][1].shape)
        for i in range(N_train):
            X = train_loader[0][i].unsqueeze(0) # Add an additional dimension to the tensor denoting the number of image data contained
            y = train_loader[1][i]
            print(y)
            optimizer.zero_grad()
            z = model.forward(X)
            z = z.view(1, -1)
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.item()
        # print(COST)
        # cost_list.append(COST / N_train)
        # correct = 0
        # N_val = len(val_loader[0])
        # # perform a prediction on the validation data
        # with
train_model(n_epochs)