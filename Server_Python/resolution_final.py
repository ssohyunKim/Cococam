#!/usr/bin/env python
# coding: utf-8

# # UpScale Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[15]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2

from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_reduce
from Subpixel import Subpixel

import coco
import utils
import model as modellib
import visualize


# 프로젝트의 디렉토리 위치를 반환 
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 웨이트값이 저장된 모델 위치
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# 이미지를 저장할 디렉토리 
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_OUT = os.path.join(ROOT_DIR, "output")
IMAGE_NPY = os.path.join(IMAGE_DI, "upTest")
Zoom = 1

# In[ ]:



while Zoom != 1
#filename = sys.agrv[1]
    filename = 'original.jpg'
    image = cv2.imread(os.path.join(IMAGE_DIR, filename))

    crop = cv2.resize(image, dsize=(44, 44)) 
    # size를 44 x 44 로 조절함 
    norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

    np.save(os.path.join(IMAGE_DIR, 'upTest','test.npy'), norm)

    #모델 생성하고 Weight 값 불러오기 

    inputs = Input(shape=(44, 44, 3))

    net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(filters=upscale_factor**2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)
    outputs = Activation('relu')(net)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse')

    model.summary()

    # load weight
    model.load_weights("model.h5")

    x1_test = np.load(os.path.join(IMAGE_NPY, filenmae))
    y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

    y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)
    y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)
    y_pred = cv2.resize(y_pred, dsize=(176, 176))
    matplotlib.image.imsave(os.path.join(IMAGE_DIR, filename), y_pred)


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[16]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()


# ## Create Model and Load Trained Weights

# In[18]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

print("finish")


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[10]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# # Run Object Detection

# In[11]:


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#filename = sys.argv[1]
filename = 'test1.jpg'
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))


# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]


    
    


# ## Person Select From Mask

# In[12]:


#마스
masks = r['masks'][:,:,r['class_ids']==1]
print(masks.shape)

b_p = np.sum(masks,axis=0)
b_p = np.sum(b_p,axis=0)

index = b_p.argmax()
for i in range(masks.shape[2]):
   if(i != index):
      masks[:,:,i] = np.zeros((masks.shape[0],masks.shape[1]))
        

mask = np.sum(masks,axis =2).astype(np.bool)
#3차원으로 바꿔줌
mask_3d = np.repeat(np.expand_dims(mask,axis=2),3,axis=2).astype(np.uint8)


# ## Blur

# In[13]:


# 샤프닝처리
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0 #정규화위해 8로나눔
image = cv2.filter2D(image.astype(np.uint8),-1,kernel_sharpen_3)
image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

blurred_img = cv2.GaussianBlur(image, (25, 25), 0)

# 마스크 블러 
mask_3d_blurred = (cv2.GaussianBlur(mask_3d*255, (101, 101), 25, 25) / 255).astype(np.float32)
#mask_3d_blurred = (cv2.medianBlur(mask_3d*255, (101))).astype(np.float32)
#mask_3d_blurred = (cv2.blur(mask_3d*255, (101, 101),0)).astype(np.float32)
# out = np.where(mask_3d, image, blurred_img)

# 합치기
person_mask = mask_3d_blurred * image.astype(np.float32)
bg_mask = (1 - mask_3d_blurred) * blurred_img.astype(np.float32)
out = (person_mask + bg_mask).astype(np.uint8)

# save images
skimage.io.imsave(os.path.join(IMAGE_OUT,filename), out)



# In[ ]:




