# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:19:25 2023

WBC cells segmentation / classification based on qubvel segmentation_models 
U-net architechture, with ResNet34 backbone with pre-trained weights for faster and better convergence
https://github.com/qubvel/segmentation_models

Multiclass U-net cells labeling:
Background      0
basophil        1
eosinophil      2
erythroblast    3
I-granulocytes  4
lymphocyte      5
monocyte        6
neutrophil      7
platelet        8

@author: Sergey Sasnouski
Project Datascientest
"""

import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from keras import models
from keras import optimizers as kopt
import segmentation_models as sm
from keras.metrics import MeanIoU
os.environ["SM_FRAMEWORK"] = "tf.keras" # 1
from keras.utils import to_categorical  # 2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD


# Part I. Read images + masks
# ***********************************************
n_classes = 2 # 3200 images Number of classes for Unet segmentation
Image_path = sorted(glob.glob('train images CentCrop 352 aug/*.jpg'))
Mask_path_bin = sorted(glob.glob('mask binary CentCrop 352 aug/*.png'))

n_classes = 9 # 800 images
Image_path = sorted(glob.glob('train images CentCrop 352 aug4/*.jpg'))
Mask_path_multi = sorted(glob.glob('mask multiclass CentCrop 352 aug4/*.png'))


train_masks = []
for mask_path in Mask_path_bin:
    mask = cv2.imread(mask_path, 0)
    train_masks.append(mask)
train_masks = np.array(train_masks)  # train_masks[0:800]

train_images = []
for img_path in Image_path:
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_images.append(img)
train_images = np.array(train_images)
train_images.shape, train_masks.shape, np.unique(train_masks)



# Part II. Transform masks
# ***********************************************
labelencoder = LabelEncoder()
n, h, w = train_masks.shape  # (1600, 128, 128), array([1, 2, 3, 4])
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3) 


# Part III. Train test split
# ***********************************************
# Train - 60%, test = 20%, validation = 20%
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.2, random_state = 5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5)

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))  

val_masks_cat = to_categorical(y_val, num_classes=n_classes) 
y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))  


# Part IV. Set Metrics + Loss + Optimiser
# ***********************************************
activation = 'sigmoid' if n_classes == 2 else 'softmax'
LR = 0.0001
optim = kopt.Adam(LR)
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 2 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


# Part V. Train model
# ***********************************************
BACKBONE1 = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train_pp = preprocess_input(X_train)
X_val_pp = preprocess_input(X_val)    

model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)
# print(model1.summary())

model1.compile(optim, total_loss, metrics=metrics)

n_epochs = 20
history1 = model1.fit(X_train_pp,
          y_train_cat,
          batch_size=8,
          epochs=n_epochs,
          verbose=1,
          validation_data=(X_val_pp, y_val_cat))

# Save model
model1.save('C://Data//!DST//Models//ImRes352_Resnet34_Epochs20_LR0001_multi_800-2.hdf5')
model1.save('C://Data//!DST//Models//Segm2_SGD_ResNet34_Epochs30_Batch8_DS3200//model', save_format='tf')
# model1 = models.load_model('C:/Data/!DST\Models/!!!ImRes352_Resnet34_Epochs20_LR0001_binary_valid2/ImRes352_Resnet34_Epochs20_LR0001_binary_valid2.hdf5', compile=False)

# Save history
np.save('C://Data//!DST//Models//Segm2_SGD_ResNet34_Epochs30_Batch8_DS3200_history.npy',history1.history)
# history=np.load('my_history.npy',allow_pickle='TRUE').item()

# Save Test data to calculate MeanIoU from another environment as Tensorflow 2.10 MeanIoU function is broken
X_test_pp = preprocess_input(X_test)
y_test_pred = model1.predict(X_test_pp)
y_test_pred_argmax = np.argmax(y_test_pred, axis=3)
np.save('C://Data//!DST//Models//y_test.npy', y_test)
np.save('C://Data//!DST//Models//y_test_pred_argmax.npy', y_test_pred_argmax)


# Part VI. Plot iou_score + loss at each epoch
# ***********************************************
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

iou = history1.history['iou_score']
val_iou = history1.history['val_iou_score']

plt.plot(epochs, iou, 'y', label='Training IOU')
plt.plot(epochs, val_iou, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()



# Part VII. Show MeanIoU for Binary segmentation (Calculatesd with TF 2.12)
# ***********************************************
# Using built in keras function MeanIoU from keras.metrics
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_test_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

# IOUs 2 classes model (masks binary)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[1,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)



# Part VIII. Show MeanIoU for Multiclass segmentation (Calculatesd with TF 2.12)
# ***********************************************
# Use preprocess and argmax to treat Test data
X_test_pp = preprocess_input(X_test)
y_test_pred = model1.predict(X_test_pp)
y_test_pred_argmax = np.argmax(y_test_pred, axis=3)

# Using built in keras function MeanIoU from keras.metrics
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_test_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

# IOUs for multi-classes model
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[0,5] + values[0,6] + values[0,7] + values[0,8] + values[1,0] + values[2,0] + values[3,0] + values[4,0] + values[5,0] + values[6,0] + values[7,0] + values[8,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[1,5] + values[1,6] + values[1,7] + values[1,8] + values[0,1] + values[2,1] + values[3,1] + values[4,1] + values[5,1] + values[6,1] + values[7,1] + values[8,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[2,5] + values[2,6] + values[2,7] + values[2,8] + values[0,2] + values[1,2] + values[3,2] + values[4,2] + values[5,2] + values[6,2] + values[7,2] + values[8,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[3,5] + values[3,6] + values[3,7] + values[3,8] + values[0,3] + values[1,3] + values[2,3] + values[4,3] + values[5,3] + values[6,3] + values[7,3] + values[8,3])
class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[4,5] + values[4,6] + values[4,7] + values[4,8] + values[0,4] + values[1,4] + values[2,4] + values[3,4] + values[5,4] + values[6,4] + values[7,4] + values[8,4])
class6_IoU = values[5,5]/(values[5,5] + values[5,0] + values[5,1] + values[5,2] + values[5,3] + values[5,4] + values[5,6] + values[5,7] + values[5,8] + values[0,5] + values[1,5] + values[2,5] + values[3,5] + values[4,5] + values[6,5] + values[7,5] + values[8,5])
class7_IoU = values[6,6]/(values[6,6] + values[6,0] + values[6,1] + values[6,2] + values[6,3] + values[6,4] + values[6,5] + values[6,7] + values[6,8] + values[0,6] + values[1,6] + values[2,6] + values[3,6] + values[4,6] + values[5,6] + values[7,6] + values[8,6])
class8_IoU = values[7,7]/(values[7,7] + values[7,0] + values[7,1] + values[7,2] + values[7,3] + values[7,4] + values[7,5] + values[7,6] + values[7,8] + values[0,7] + values[1,7] + values[2,7] + values[3,7] + values[4,7] + values[5,7] + values[6,7] + values[8,7])
class9_IoU = values[8,8]/(values[8,8] + values[8,0] + values[8,1] + values[8,2] + values[8,3] + values[8,4] + values[8,5] + values[8,6] + values[8,7] + values[0,8] + values[1,8] + values[2,8] + values[3,8] + values[4,8] + values[5,8] + values[6,8] + values[7,8])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)
print("IoU for class6 is: ", class6_IoU)
print("IoU for class7 is: ", class7_IoU)
print("IoU for class8 is: ", class8_IoU)
print("IoU for class9 is: ", class9_IoU)


# Part IX. Show Test-image + Test-mask + Prediction-mask 
# ***********************************************
test_img_number = 1
test_img = X_test_pp[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
test_img_input1 = preprocess_input(test_img_input)
test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]
mask_diff = cv2.subtract(ground_truth, test_prediction1)

# *************************
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
plt.imshow(test_img)

plt.subplot(232)
plt.title('Manual Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray', interpolation='none')

plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1, cmap='gray', interpolation='none')
plt.show()

# Show ground_truth mask + ground_truth-predisted mask
# ***********************************************
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.title('Manual Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray', interpolation='none')

plt.subplot(222)
plt.title('Manual Testing Label')
plt.imshow(mask_diff, interpolation='none')
plt.show()
