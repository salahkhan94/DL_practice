from torch.nn import Module, Conv2d, Linear, AdaptiveAvgPool2d
from torch.nn.functional import relu, sigmoid
import numpy as np
import cv2
import os
import torch
import random
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

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
    y = y.flatten()
    # print('test', X.shape)
    # y1 = torch.tensor(1)
    # y = torch.add(y, y1)

    return X, y

img_files = os.listdir('/home/salahuddin/projects/Deeplearning_Practice/mix/')
img_files = list(filter(lambda x: x != 'train', img_files))
def train_path(p): return f"/home/salahuddin/projects/Deeplearning_Practice/mix//{p}"
img_files = list(map(train_path, img_files))

print("total training images", len(img_files))
print("First item", img_files[0])

random.shuffle(img_files)

train = img_files[:20000]
test = img_files[20000:]

print("train size", len(train))
print("test size", len(test))

# image normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# preprocessing of images
class CatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self): return self.len

    def __getitem__(self, index): 
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if 'cat' in path else 1
        return (image, label)

# create train dataset
train_ds = CatDogDataset(train, transform)
train_dl = DataLoader(train_ds, batch_size=100)
print(len(train_ds), len(train_dl))

# create test dataset
test_ds = CatDogDataset(test, transform)
test_dl = DataLoader(test_ds, batch_size=100)
print(len(test_ds), len(test_dl))


# # image normalization
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

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
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

model = Network()
# for param in model.parameters():
#     torch.nn.init.uniform_(param, -1, 1)

# for name, param in model.named_parameters():
#     print(name, param.data)

# model.load_state_dict(torch.load("models/test_model.pt"))
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_loader = load_data(data_dir=data_dir)
N_train = len(train_loader[0])
n_epochs = 10

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        # print(train_loader[0][1].shape)
        for X, y in train_dl:
            z = model.forward(X)
            optimizer.zero_grad()
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
