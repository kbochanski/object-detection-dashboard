import logging
import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
#import tensorflow as tf
#import tensorflow_hub as hub
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys


from .constants import YOLOV3_CONFIG_PATH,YOLOV3_WEIGHTS_PATH


logger = logging.getLogger(__name__)

def show_image(image,save = False):
    img = Image.fromarray(image, 'RGB')
    if save:
        img.save('output.png')
    img.show()

def get_file_path_from_current_file(path_from_current_file):
     return os.path.join(os.path.dirname(__file__),path_from_current_file) 

def read_image(path:str) -> np.array:
        """Read image and return as numpy array
        """
        if path != None:
            image_object = Image.open(path)
            image_array = np.array(image_object)
        else:
            return ValueError('Image could not be processed')
        return image_array
        
def get_object_detection_label_names():
        """Get label names from Github
        """
        url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
        f = urllib.request.urlopen(url)
        classNames = [line.decode('utf-8').strip() for  line in f]
        return classNames

def read_and_configure_darknet(config_path:str,weights_path:str):
    """Read in neural network.

    [Darknet](https://pjreddie.com/darknet/) is a repository for
    Open Source Neural Networks in C. Function part of
    [OpenCV](https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gafde362956af949cce087f3f25c6aff0d)
    """

    # Read
    net = cv2.dnn.readNetFromDarknet(
             get_file_path_from_current_file(config_path),
               get_file_path_from_current_file(weights_path),
               )
    # Configure
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def extract_model_info(net,image:np.ndarray):
    """Run image through prediction layer of neural network

    Find out details of blobFromImage [here](https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
    and [function docs](https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7)
    """

    logger.info(f'Image Dimensions: {image.shape}')
    blob = cv2.dnn.blobFromImage(image, scalefactor= 1 / 255, size=(320,320), mean=[0, 0, 0],swapRB=False,crop=False)
    
    # Sets the new input value for the network.
    net.setInput(blob)
    layersNames = net.getLayerNames()
    logger.info(f'Network has {len(layersNames)} layers')

    # Runs forward pass to compute output of layer using outputNames
    outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    return outputs

def calc_bounding_boxes(image:np.ndarray,predictions,confidence_threshold = 50, nms_threshold = 20):
    """Calculate bounding box coordinates, classes and  class probabilities

    Args:
        predictions: Object estimates from neural net
        image: Image
        confidence_threshold: = 50
        nms_threshold: = 20
    """

    # (height, width, depth) where depth is RGB colors thus should be 3
    h, w, d = image.shape
     
    bbox = []
    class_ids = []
    confs = []
    for output in predictions:
        # What is det?
        for det in output:
            # logger.info(det)
            scores = det[5:]
            # logger.info(scores)
            class_id = np.argmax(scores)
            # logger.info(class_id)
            confidence = scores[class_id]
            if confidence > (confidence_threshold/100):
                width,height = int(det[2]*w) , int(det[3]*h)
                x,y = int((det[0]*w)-width/2) , int((det[1]*h)-height/2)
                bbox.append([x,y,width,height])
                class_ids.append(class_id)
                confs.append(float(confidence))
    
    # logger.info(bbox) # List of bounding boxes (4 coords)
    # logger.info(classIds) # List of predicted classes (by id)
    # logger.info(confs) # Probability of each class

    return bbox, class_ids, confs


def draw_bounding_boxes(image:np.ndarray,predictions, confidence_threshold , nms_threshold):
    """Use cv2 library to draw bounding boxes on image
    """

    # Get class names
    class_names = get_object_detection_label_names()

    # Calculate bounding boxes
    bbox, class_ids, confs = calc_bounding_boxes(image, predictions,confidence_threshold, nms_threshold)
    
    # Performs non maximum suppression given boxes and corresponding scores.
    # For each bounding box, the network also predicts the confidence that the bounding box actually encloses an object, 
    # and the probability of the enclosed object being a particular class.
    # Most of these bounding boxes are eliminated because their confidence is low 
    # or because they are enclosing the same object as another bounding box with a very high confidence score. 
    # This technique is called non-maximum suppression.
    # https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidence_threshold/100, nms_threshold/100)

    obj_list=[]
    confi_list =[]
    # Draw rectangle around object after removing noise/low probability boxes
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # logger.info((x,y,w,h))
        cv2.rectangle(image, (x, y), (x+w,y+h), (240, 54 , 230), 2)
        # logger.info((i,confs[i],class_ids[i]))
        obj_list.append(class_names[class_ids[i]].upper())
        
        confi_list.append(int(confs[i]*100))
        cv2.putText(image,f'{class_names[class_ids[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
        

    # Lists to dataframe
    df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
    
    # s = Image.fromarray(image, 'RGB').size
    # logger.info(s)
    # show_image(image)
    return image,obj_list,confi_list,df


