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

from pages.common.constants import PAGES_PATH
from pages.image_detection.main import read_image,read_and_configure_darknet,extract_model_info,draw_bounding_boxes
from pages.image_detection.constants import YOLOV3_CONFIG_PATH,YOLOV3_WEIGHTS_PATH


def object_detection_image():

    # UploadedFile class is a subclass of BytesIO,
    uploaded_file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])

    # For debugging (uses file in pages DIR)
    # uploaded_file = 'app/pages/cat-party.png'

    if uploaded_file:
        raw_image = read_image(uploaded_file)
        st.image(raw_image, caption = "Uploaded Image")
    
        # Image processing settings
        confidence_threshold =st.slider('Confidence', 0, 100, 50)
        nms_threshold= st.slider('Threshold', 0, 100, 20)
        show_ci_levels = st.checkbox("Show Confidence Levels",value=True )

        process_image = st.button('Process Image')
        if process_image:
            my_bar = st.progress(0)
            net = read_and_configure_darknet(YOLOV3_CONFIG_PATH,YOLOV3_WEIGHTS_PATH)
            predictions = extract_model_info(net,raw_image)
            image,obj_list,confi_list,df = draw_bounding_boxes(raw_image,predictions, confidence_threshold, nms_threshold)
            my_bar.progress(100)
            
            st.image(image, caption='Processed Image.')
            if show_ci_levels: 
                st.subheader('Confidence levels')
                st.write(df)
                # st.bar_chart(df["Confidence"])

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()





def main():
    st.title('Object Detection for Images')
    st.subheader("YoloV3")

    st.markdown("""
    This project demonstrates YOLO Object detection in images.
    This object detection project takes in an image and outputs the image with bounding boxes 
    created around the objects in the image.
    
    This YOLO object Detection project can detect 80 objects(i.e classes)
    in either a video or image. The full list of the classes can be found 
    [here](https://github.com/KaranJagtiani/YOLO-Coco-Dataset-Custom-Classes-Extractor/blob/main/classes.txt)"""
    )
    object_detection_image()


main()