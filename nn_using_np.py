from torch.nn import Module, Conv2d, Linear, AdaptiveAvgPool2d
from torch.nn.functional import relu, sigmoid
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        super().__init__()

        # onvolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # conected layers
        self.fc1 = nn.Linear(in_features= 64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)


    def forward(self, X):

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        # print(X.shape)    
        # print("h1")
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        # print("h2")
        # print(X.shape)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)
        # print("h3")
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        # print(X.shape)                                              
        # print("h5")
        X = F.relu(self.fc2(X))
        # print(X.shape)
        # print("h6")
        X = self.fc3(X)
        # print(X.shape)
        # print("h7")

        return X
model = Network()
for param in model.parameters():
    torch.nn.init.uniform_(param, -1, 1)

# for name, param in model.named_parameters():
#     print(name, param.data)

# model.load_state_dict(torch.load("models/test_model.pt"))
learning_rate = 0.01
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_loader = load_data(data_dir=data_dir)
N_train = len(train_loader[0])
n_epochs = 1

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        # print(train_loader[0][1].shape)
        for i in range(N_train):
            X = train_loader[0][i].unsqueeze(0) # Add an additional dimension to the tensor denoting the number of image data contained
            y = train_loader[1][i]
            print(y)
            z = model.forward(X)
            # print(z.shape)
            z = z.view(1, -1)
            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            print(z.shape, y.shape)
            loss = criterion(z, y)
            loss.backward() 
            
            optimizer.step()
            COST += loss.item()
            print(" ", loss.item())
            print("")
        print(COST)
        # cost_list.append(COST / N_train)
        # correct = 0
        # N_val = len(val_loader[0])
        # # perform a prediction on the validation data
        # with
train_model(n_epochs)

# def test_model():

# torch.save(model.state_dict(), "/home/salahuddin/projects/Deeplearning_Practice/test_model.pt")
# mix_data_dir = '/home/salahuddin/projects/Deeplearning_Practice/test_mix'
# test_loader = load_data(data_dir=mix_data_dir)

# def testmodel :
