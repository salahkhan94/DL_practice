import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
data_name = pd.read_csv('index.csv')
# Create your own dataset object
class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        data_dircsv_file=os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = Dataset(csv_file='index.csv' , data_dir='',transform=transforms.CenterCrop(20) )
# print("The shape of the first element tensor: ", dataset[0][0].shape)

image = dataset[10][0]
print(dataset[10][1])
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.show()



def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0], cmap='gray')
    plt.title('y = ' + data_sample[1])
    plt.show()
# Create your own dataset object
show_data(dataset[0],shape = (20, 20))