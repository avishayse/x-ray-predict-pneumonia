#!/usr/bin/env python
# coding: utf-8

# # Binary classification with Keras neural network

# This notebook can be used if you want to train the model yourself!
# 
# Original notebook: https://www.kaggle.com/kosovanolexandr/keras-nn-x-ray-predict-pneumonia-86-54
# 
# Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# ### Imports

# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

import os
import numpy as np
import pandas as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Verify our directories structure

# In[5]:


print(os.listdir("/data/xray-images/chest_xray")) 

print(os.listdir("/data/xray-images/chest_xray/test"))

print(os.listdir("/data/xray-images/chest_xray/train/"))

print(os.listdir("/data/xray-images/chest_xray/val/"))


# ### Check an image in the "NORMAL" training set

# In[6]:


img_name = 'NORMAL2-IM-0588-0001.jpeg'
img_normal = load_img('/data/xray-images/chest_xray/train/NORMAL/' + img_name)

print('NORMAL')
plt.imshow(img_normal)
plt.show()


# ### Check an image in the PNEUMONIA training set

# In[7]:


img_name = 'person63_bacteria_306.jpeg'
img_pneumonia = load_img('/data/xray-images/chest_xray/train/PNEUMONIA/' + img_name)

print('PNEUMONIA')
plt.imshow(img_pneumonia)
plt.show()


# ### Initialize variables

# In[8]:


# dimensions of our images.
img_width, img_height = 150, 150


# In[21]:


train_data_dir = '/data/xray-images/chest_xray/train'
validation_data_dir = '/data/xray-images/chest_xray/val'
test_data_dir = '/data/xray-images/chest_xray/test'

nb_train_samples = 5217
nb_validation_samples = 17
epochs = 20
batch_size = 16


# In[10]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# ### Create Sequential model

# In[11]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# ### Check information about the model

# In[12]:


model.layers


# In[13]:


model.input


# In[14]:


model.output


# ### Compile the model

# In[15]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# ### Upload images from the different sets

# In[16]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[17]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[18]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[22]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[23]:


test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# ### Fit the model

# In[25]:


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# ### Save the model (weights + complete model)

# In[28]:


model.save_weights('output/first_try.h5')


# In[29]:


model.save('output/pneumonia_model.h5')

