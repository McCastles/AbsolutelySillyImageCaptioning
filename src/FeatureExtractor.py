#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.



import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# print('ROOT_DIR =', ROOT_DIR)
# print(sys.path)


# # Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# # Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# print(sys.path)
import coco


get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")




# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# COCO_MODEL_PATH




# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print('nope')
#     utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights



# import tensorflow
# print(tensorflow.__version__)
# stop




# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs,
# which are integer value that identify each class.
# Some datasets assign integer values to their classes and some don't.
# The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. 
# The IDs are often sequential, but not always.
# 
# To improve consistency, and to support training on data from multiple sources at the same time,
# our ```Dataset``` class assigns it's own sequential integer IDs to each class.
# For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO)
# and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
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
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below.
# The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)



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
               'teddy bear', 'hair drier', 'toothbrush'] #, 'pillow'


# ## Run Object Detection








def extract(img_path, show=True):

    # if URL:
        # get image from URL
    image = skimage.io.imread(img_path)

    # original image
    if show:
        plt.figure(figsize=(12,10))
        skimage.io.imshow(image)


    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])




    # print(results[0].keys())
    # for i in results[0].values():
    #     print(i, '\n')




    class_ids = r['class_ids']
    # print(class_ids)

    mask = r['masks']
    mask = mask.astype(int)
    # mask.shape

    show_segments = False

    if show_segments:
        for i in range(mask.shape[2]):
            temp = skimage.io.imread(img_path)
            for j in range(temp.shape[2]):
                temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
            plt.figure(figsize=(8,8))
            
            class_id = class_ids[i]
            class_name = class_names[ class_id ]
            plt.title(class_name)
            plt.imshow(temp)


    
    labels = []
    bbox_ids = []
    # for bbox_id, i in enumerate(1, class_ids):
    for i in range(len(class_ids)):

        class_id = class_ids[i]

        bbox_ids.append( i+1 )
        labels.append( class_names[ class_id ] )
        
        print(i, class_id, class_names[class_id])

    r['bbox_ids'] = bbox_ids
    r['image'] = image
    r['labels'] = labels

    return r