import os

import numpy as np
import pickle 

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

'''Support Vector Machines'''

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
labels = np.asarray(labels)


'''train / test split'''

#split data into train and test set || y = labels
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#stratify --> stratified sampling: keeping the same proportions of "categories" in the sample as in all the data

'''train classifier'''

#classifier choice: using default values for SVC
classifier = SVC()

#training multiple image classifiers using each combination of gamma and C (12 combos)
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

#use GridSearchCV to train multiple classifiers at once
grid_search = GridSearchCV(classifier, parameters)

#start training
grid_search.fit(x_train, y_train)


'''test performance'''

#get the best of all 12 trained image classifiers
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test) #model.predict() --> given a trained model, predict the label of a new set of data

score = accuracy_score(y_test, y_prediction) #compute accuracy of set of labels predicted for the sample

print('{}% of samples were correctly classified'.format(str(score * 100)))

#save best trained model into a file
pickle.dump(best_estimator, open('./model.p', 'wb'))
