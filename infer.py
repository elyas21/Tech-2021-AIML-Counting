# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:11:52 2018

@author: lenovo
"""
import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json
import cv2
import time
import os

def load_model():
    # Function to load and return neural network model 
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model

def create_img(path):
    #Function to load,normalize and return image 
    print(path)
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    return im

def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = load_model()#加载模型
    image = create_img(path)#图像预处理
    ans =   model.predict(image)#预测数据
    count = np.sum(ans)
    return count,image,ans

"""
ans,img,hmap = predict('ShanghaiTech/test_data/test_A_36.jpg')
print("Predict Count:",ans)
#Print count, image, heat map
plt.imshow(img.reshape(img.shape[1],img.shape[2],img.shape[3]))
plt.show()
plt.imshow(hmap.reshape(hmap.shape[1],hmap.shape[2]) , cmap = c.jet )
plt.show()
"""

def getPridiction(image):
    ans,img,hmap = predict('test_images/myimg/' + image)
    print("Predict Count:",ans)
    #Print count, image, heat map
    plt.imshow(img.reshape(img.shape[1],img.shape[2],img.shape[3]))
    plt.show()
    plt.imshow(hmap.reshape(hmap.shape[1],hmap.shape[2]) , cmap = c.jet )
    plt.show()
    temp = h5py.File('ShanghaiTech/part_A/test_data/ground_truth/IMG_170.h5' , 'r')
    temp_1 = np.asarray(temp['density'])
    #plt.imshow(temp_1,cmap = c.jet)
    print("Original Count : ",int(np.sum(temp_1)) + 1)

# Video source - can be camera index number given by 'ls /dev/video*
# or can be a video file, e.g. '~/Video.avi'
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     img_counter = 200
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     while(True):
#         img_name = "IMG_{}.png".format(img_counter)
#         path = 'ShanghaiTech/part_A/test_data/images/'
#         cv2.imwrite(os.path.join(path , img_name), frame)
#         getPridiction(img_name)
#         img_counter += 1
#         time.sleep(10)
        
    

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

getPridiction('000.jpg')