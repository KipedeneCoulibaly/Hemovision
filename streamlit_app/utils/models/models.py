import os
import cv2
import numpy as np
import tensorflow as tf
from keras import models
os.environ["SM_FRAMEWORK"] = "tf.keras"
from matplotlib import pyplot as plt
import segmentation_models as sm
import streamlit as st

#@st.cache_data
def img_processing(img_path):
    """Preprocess image"""
    SIZE_X = 352  
    SIZE_Y = 352
    images = []
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resx, resy, _ = img.shape
    if (resx != SIZE_X) or (resy != SIZE_Y):
        img = cv2.resize(img, (SIZE_X, SIZE_Y))
    images.append(img)
    return np.array(images)

def show_img_mask(images, img_mask_pred, i):
    """Plot images and its mask"""
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.title('Testing Image')
    plt.imshow(images[i])

    plt.subplot(222)
    plt.title('Prediction on Testing image')
    plt.imshow(img_mask_pred[i], cmap='gray', interpolation='none')
    plt.show()

@st.cache_resource()
def Hemovision(img_path, show_mask=False):
    """
    Predicts image class
    parameters :
        img_path  : image path.
        show_mask : plot images and its mask
    return : classes predicted by the models.
    """     
    images = img_processing(img_path)
    # Load segmentation U-net model
    model_segm = models.load_model("streamlit_app/data/model/Seg_ResNet34.hdf5", compile=False)
    # preprocess input image and predict its mask
    preprocess_input = sm.get_preprocessing('resnet34')
    X_test_pp = preprocess_input(images)
    y_test_pred = model_segm.predict(X_test_pp)
    img_mask_pred = np.argmax(y_test_pred, axis=3)
    # Choose Image number
    i = 0
    if i > (len(images)-1):
        i = len(images)-1
    # Show resulting images
    if show_mask:
        show_img_mask(images, img_mask_pred, i)

    # Classify RGB images
    # ***********************************************
    n_classes  = 8
    classabr8  = {0:'BA',1:'NE',2:'EO',3:'ERB',4:'IG',5:'LY',6:'MO',7:'PLT'}
    classname8 = {0:'Basophil',1:'Neutrophil',2:'Eosinophil',3:'Erythroblast',
                    4:'Immature Granulocyte',5:'Lymphocyte',6:'Monocyte',7:'Platelet'}

    # ***********************************************
    n_classes = 11
    classabr11  = {0:'BA',1:'BNE',2:'EO',3:'ERB',4:'LY',5:'MMY',
                    6: 'MO',7:'MY',8:'PLT',9:'PMY',10:'SNE',}
    classname11 = {0:'Basophil',1:'Band Neutrophil',2:'Eosinophil',3:'Erythroblast',
                    4:'Lymphocyte',5:'Meta-Myelocytes',6:'Monocyte',7:'Myelocytes',
                    8:'Platelet',9:'Pro-Myelocytes',10:'Segmented Neutrophil',}

    # Multiply image by its mask
    images_crop = []
    for i, img in enumerate(images):
        gray_three = cv2.merge([img_mask_pred[i],img_mask_pred[i],img_mask_pred[i]])
        crop = np.multiply(img, gray_three)
        images_crop.append(crop)
    images_crop = np.array(images_crop)
    # plt.imshow(images_crop[0])

    model_class8  = tf.keras.models.load_model('streamlit_app/data/model/ResNet34_8c.hdf5') 
    model_class11 = tf.keras.models.load_model('streamlit_app/data/model/ResNet34_11c.hdf5')
        
    
    return model_class8, classname8, model_class11, classname11, images_crop
