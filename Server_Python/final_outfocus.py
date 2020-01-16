#!/usr/bin/env python
# coding: utf-8


# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2

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


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


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

# In[4]:


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

# In[5]:


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
while True:

#	filename = sys.argv[1]
    a = input()
    filename = 'original.jpg'
#filename = 'test1.jpg'
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))


# Run detection
    results = model.detect([image], verbose=1)

# Visualize results
    r = results[0]


    
    


# ## Person Select From Mask

# In[6]:


#마스
    masks = r['masks'][:,:,r['class_ids']==1]

    b_p = np.sum(masks,axis=0)
    b_p = np.sum(b_p,axis=0)
#print(b_p)

    index = b_p.argmax()
    for i in range(masks.shape[2]):
        if(i != index):
            masks[:,:,i] = np.zeros((masks.shape[0],masks.shape[1]))
        

    mask = np.sum(masks,axis =2).astype(np.bool)
#3차원으로 바꿔줌
    mask_3d = np.repeat(np.expand_dims(mask,axis=2),3,axis=2).astype(np.uint8)


# ## Blur

# In[7]:


# 원본이미지 블러
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

#회전
    height, width, channel = out.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), -90, 1)
    out = cv2.warpAffine(out, matrix, (width, height))

# save images
    skimage.io.imsave(os.path.join(IMAGE_OUT,filename), out)
    print("save finish")



# In[ ]:




