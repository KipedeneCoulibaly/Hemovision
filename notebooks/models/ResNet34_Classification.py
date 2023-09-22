# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:19:25 2023

WBC cells ResNet34 classification based on qubvel classification_models 
https://github.com/qubvel/classification_models

@author: Sergey Sasnouski
Project Datascientest
"""

import numpy as np
import pandas as pd
import cv2
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from classification_models.tfkeras import Classifiers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
# from tensorflow.keras import preprocessing
from tensorflow.keras import models


# Part I. Define 8 or 11 classes model
# ***********************************************
# label images dataset
n_classes = 8
labels = ['BA', 'NE', 'EO', 'ERB', 'IG', 'LY', 'MO', 'PLT']
classes = ['Basophil', 'Neutrophil', 'Eosinophil', 'Erythroblast', 'IGranulocytes', 'Lymphocyte', 'Monocyte', 'Platelet']            
path = 'C:\\Data\\!DST\\8classes_filter_xmask_gtrain\\'
Gpath = 'C:\\Data\\!DST\\8classes_filter_xmask_gtest\\'


n_classes = 11
labels = ['BA', 'BNE', 'EO', 'ERB', 'LY', 'MMY', 'MO', 'MY', 'PLT', 'PMY', 'SNE']
classes = ['Basophil', 'Band Neutrophil', 'Eosinophil', 'Erythroblast', 'Lymphocyte', 'Meta-Myelocytes', 'Monocyte', 'Myelocytes', 'Platelet', 'Pro-Myelocytes', 'Segmented Neutrophils']
path = 'C:\\Data\\!DST\\11classes_filter_xmask_gtrain\\'
Gpath = 'C:\\Data\\!DST\\11classes_filter_xmask_gtest\\'


# Part II. Read Train / Validation images and labels
# ***********************************************
DS1_img = []
DS1_labels = []
DS1_classes = []
for i, label in enumerate(labels):
    path_label = os.path.join(path, label)
    for img in os.listdir(path_label):
        img = cv2.imread(os.path.join(path_label, img), 1)
        img = img[:,:,[2,1,0]]  # BGR to RGB
        img = cv2.resize(img, (176, 176))
        DS1_img.append(img)
        DS1_labels.append(labels.index(label))
        DS1_classes.append(classes[i])
DS1_img = np.array(DS1_img)
DS1_labels = np.array(DS1_labels)


# Read Test images and labels
# **************************************
X_test = []
y_test = []
DS1_classes = []
for i, label in enumerate(labels):
    path_label = os.path.join(Gpath, label)
    for img in os.listdir(path_label):
        img = cv2.imread(os.path.join(path_label, img), 1)
        img = img[:,:,[2,1,0]]  # BGR to RGB
        img = cv2.resize(img, (176, 176))
        X_test.append(img)
        y_test.append(labels.index(label))
        DS1_classes.append(classes[i])
X_test = np.array(X_test)
y_test = np.array(y_test)


# Plot images Labels
# **************************************
df_labels = pd.DataFrame(
    # {'Label': DS1_labels,
    #  'Cell type': DS1_classes
    {'Label': y_test,
     'Cell type': DS1_classes
    })
df_labels.head()

# create countplot
plt.figure(figsize=(10,5))
chart = sns.countplot(
    data = df_labels,
    x = 'Cell type',
    palette = 'Set1',
)
chart.set(xlabel=None)  # remove the axis label

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)

plt.ylabel('Images', fontsize='x-large')
plt.title('Images par type cellulaire', fontsize='x-large')
# **************************************


# Part III. Train test split / One-hot encoding
# **************************************
# Train - 90%, validation - 10%, NO test, test after - with Gtest
X_train, X_val, y_train, y_val = train_test_split(DS1_img, DS1_labels, test_size=0.1, random_state=1)

y_train = np_utils.to_categorical(y_train, n_classes)
y_val = np_utils.to_categorical(y_val, n_classes)
# y_test = np_utils.to_categorical(y_test, n_classes)


# show first 8 images
for i in range(1, 9):
    img = X_train[i-1]
    plt.subplot(2, 4, i)
    plt.imshow(img, interpolation='none')
    plt.axis('off')
    plt.title(labels[np.argmax(y_train[i-1])])
print("Shape of each image in the training data: ", X_train.shape[1:])



# Part IV. Build and compile model
# ***********************************************
ResNet34, preprocess_input = Classifiers.get('resnet34')
# base_model = ResNet34(input_shape=(352,352,3), weights='imagenet', include_top=False)
base_model = ResNet34(input_shape=(176,176,3), weights='imagenet', include_top=False)

x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
# model.summary()

opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9) # nesterov=True

filepath = 'C:\\Data\\!DST\\\Models\\best_model.epoch{epoch:02d}-acc{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", mode="max", save_best_only=True,
                             initial_value_threshold = 0.88, verbose=1)
callbacks = [checkpoint]

# compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# Part V. Train the network / save model / history
# ***********************************************
history1 = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

# saving the model in tensorflow format (for further training contunuation)
model.save('C:\\Data\\!DST\\\Models\\Class8_Res176_Resnet34_Epochs20_BC32_12500_train90\\model\\', save_format='tf')
model.save('Models\Class11_ResNet34_ImRes352_Epochs20_BC64_callbacks_3300\model\md.hdf5')

# Save history
np.save('C:\\Data\\!DST\\\Models\\Class8_Res176_Resnet34_Epochs20+20_BC32_12500_train90_Gtest\\history.npy',history1.history)
# history=np.load('my_history.npy',allow_pickle='TRUE').item()

# loading the saved model
# model = tf.keras.models.load_model('C:\\Data\\!DST\\Models\\Class8_Res176_Resnet34_Epochs20_BC32_12500_train90_Gtest\\model')
# model.evaluate(X_test,y_test)



# Part VI. Plot the training and validation accuracy and loss at each epoch
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

acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Part VII. Make predictions on Test dataset, show Classification report / Confusion matrix
# *****************************************************

y_test2 = np.argmax(y_test, axis=1)
y_pred = model.predict(X_test)
y_pred2 = np.argmax(y_pred, axis=-1)
print(classification_report(y_test2, y_pred2, target_names = classes, digits=3))

# plotting a Confusion Matrix
cm = confusion_matrix(y_test2, y_pred2)
print(cm)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=classes,
            yticklabels=classes,
    )
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix filter-out 1834',fontsize=17)
plt.show()


# Part VIII. Collecting misclassified images in test dataset
# *****************************************************

# actual = np.argmax(y_test, axis=1)
actual = y_test
prediction=[]
accuracy=[]
# TruePred=[]
incorrect_pred = []
correct_pred = []
incorrect_class = []
predictions = []
actuals = []

for idx, val in enumerate(X_test):
  val = np.expand_dims(val, axis=0)
  pred = model.predict(val)
  predicted_value = np.argmax(pred, axis=1)
  prediction.append(predicted_value)
  accuracy.append(np.amax(pred, axis=1)*100)
  if actual[idx] != predicted_value:
      incorrect_pred.append(idx)
      predictions.append(int(predicted_value))
      actuals.append(int(actual[idx]))
  else:
      correct_pred.append(idx)


# Plot 8 random images from Test dataset with real+predicted labels
ipred_idx_set = np.random.randint(low=1, high=len(incorrect_pred), size=(8,))

classname = labels
plt.figure(figsize=(12, 6))
for i in range(1, 9):
    # img = X_test[incorrect_pred[i-1]]
    ipred_idx = ipred_idx_set[i-1]
    img = X_test[incorrect_pred[ipred_idx]]
    plt.subplot(2, 4, i)
    # plt.tight_layout()
    plt.imshow(img, interpolation = 'None')
    plt.axis('off')
    plt.title("actual: %s\npredicted: %s" % (classname[actuals[ipred_idx]], classname[predictions[ipred_idx]]), fontsize=8)






