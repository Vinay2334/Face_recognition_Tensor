import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
from tensor import model

def data_for_visualization():
    Vdata = []
    for img in tqdm(os.listdir("ImageForVisualization")):
        path = os.path.join("ImageForVisualization", img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        Vdata.append([np.array(img_data), img_num])
    shuffle(Vdata)
    return Vdata
Vdata = data_for_visualization()
fig = plt.figure(figsize=(20,20))
for num, data in enumerate(Vdata[:20]):
    img_data = data[0]
    y = fig.add_subplot(5,5, num+1)
    image = img_data
    data = img_data.reshape(50,50,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0:
        my_label = 'Vinay'
    elif np.argmax(model_out) == 1:
        my_label = 'Manish'
    else:
        my_label = 'Bijay'
        
    y.imshow(image, cmap='gray')
    plt.title(my_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()