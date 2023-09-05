import numpy as np
import os
import cv2
from random import shuffle
from tqdm import tqdm
# from main.py import generate_dataset

def my_label(image_name):
    name = image_name.split('.')[-3]
    #If 2 names
    if name=="Vinay":
        return np.array([1,0])
    elif name=="Yash":
        return np.array([0,1])
# Create Data
def my_data():
    data=[]
    for img in tqdm(os.listdir("data")):
        path=os.path.join("data",img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data,(50,50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data
data = my_data()

#Training model
#80% to train remaining for testing
train = data[:400]
test = data[400:]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1) #0=X and 1=Y 
print(X_train.shape)
Y_train = [i[1] for i in train] #image in X_train and label in Y_train
X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
print(X_test.shape)
Y_test = [i[1] for i in test]
