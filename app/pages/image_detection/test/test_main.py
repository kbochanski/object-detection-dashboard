import os
import pytest
import logging
import urllib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from image_detection.main import read_image,get_object_detection_label_names,read_and_configure_darknet, extract_model_info,draw_bounding_boxes
from image_detection.constants import YOLOV3_CONFIG_PATH,YOLOV3_WEIGHTS_PATH


import pprint

logger = logging.getLogger(__name__)
logger.info('starting testing...')

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = 'data/'
TEST_IMAGE = 'cat-party.png'
TEST_IMAGE_PATH = os.path.join(TEST_DIR, TEST_DATA,TEST_IMAGE)

@pytest.fixture
def image():
    image = read_image(TEST_IMAGE_PATH)
    return image

@pytest.fixture
def net():
    net = read_and_configure_darknet(YOLOV3_CONFIG_PATH,YOLOV3_WEIGHTS_PATH)
    return net

@pytest.fixture
def predictions(image,net):
    predictions = extract_model_info(net,image)
    return predictions


def check_list_types(lst,typ):
    return all(map(lambda x: isinstance(x,typ), lst))

def test_read_image_returns_array(image):
    assert isinstance(image, np.ndarray)

def test_label_type_is_string():
    try:
        names = get_object_detection_label_names()
        assert check_list_types(names,str)
    except urllib.error.URLError:
        logger.info('No internet connection')


def test_read_darknet_model_config():
    read_and_configure_darknet(YOLOV3_CONFIG_PATH,YOLOV3_WEIGHTS_PATH)

def test_extract_darknet_model_config(net,image):
    output = extract_model_info(net,image)
    
    # Check output is tuple (of numpy arrays)
    assert isinstance(output, tuple)
    assert isinstance(output[0], np.ndarray)

def test_detect_objects(image,predictions):

    image,obj_list,confi_list,df = draw_bounding_boxes(image,predictions, 50, 20)

    expected_df = pd.DataFrame({'Object Name': ['DOG','DOG','DOG',], 'Confidence': [99,98,80,]})

    assert_frame_equal(df, expected_df)



