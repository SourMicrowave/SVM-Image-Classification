import os

import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

'''prepare data'''

input_dir = 'clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

#iterate through images: load, read, format
for category_idx, category in enumerate(categories):
  for file in os.listdir(os.path.join(input_dir, category, )): #get list of all files directory
    img_path =  os.path.join(input_dir, category, file)
    img = imread(img_path) #read image
    img = resize(img, (15, 15)) #resize (format) image
    data.append(img.flatten())#append the flattened (make into array) image into the list
    labels.append(category_idx)

    

data = np.asarray(data)
labels = np.asarray(data)



'''train / test split'''

#split data into train and test set


'''train classifier'''

'''test performance'''