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

data_name = pd.read_csv('index.csv')
print(data_name.head())
print('File name:', data_name.iloc[0, 1])
print('y:', data_name.iloc[0, 0])
image_name =data_name.iloc[1, 1]
image_path=os.path.join(image_name)
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[1, 0])
plt.show()