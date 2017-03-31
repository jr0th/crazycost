
# coding: utf-8

# In[ ]:

import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

import helper.loss_functions

import tensorflow as tf

import functools


# In[ ]:

# set up config for GPU
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "2"
session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)


# In[ ]:

dim1 = 64
dim2 = 64

batch_size = 100
epochs = 100

path_x = '../data/dummy_y3D/x.npy'
path_y = '../data/dummy_y3D/y.npy'

tb_log_dir = '../logs/'


# In[ ]:

x = keras.layers.Input((dim1, dim2, 1))

options_conv = {"activation": "relu", "kernel_size": (5, 5), "padding": "same"}
options_max_pool = {"pool_size" : (2,2), "strides" : (2,2)}

y = keras.layers.BatchNormalization()(x)

y = keras.layers.Conv2D(16, **options_conv)(y)
y = keras.layers.MaxPool2D(**options_max_pool)(y)
y = keras.layers.BatchNormalization()(y)

y = keras.layers.Conv2D(32, **options_conv)(y)
y = keras.layers.MaxPool2D(**options_max_pool)(y)
y = keras.layers.BatchNormalization()(y)

y = keras.layers.Conv2D(64, **options_conv)(y)
y = keras.layers.MaxPool2D(**options_max_pool)(y)
y = keras.layers.BatchNormalization()(y)

y = keras.layers.Conv2D(64, **options_conv)(y)

y = keras.layers.UpSampling2D()(y)
y = keras.layers.BatchNormalization()(y)
y = keras.layers.Conv2D(32, **options_conv)(y)

y = keras.layers.UpSampling2D()(y)
y = keras.layers.BatchNormalization()(y)
y = keras.layers.Conv2D(16, **options_conv)(y)

y = keras.layers.UpSampling2D()(y)
y = keras.layers.BatchNormalization()(y)
y = keras.layers.Conv2D(8, **options_conv)(y)

y = keras.layers.Conv2D(3, **options_conv)(y)

y = keras.layers.Activation('softmax')(y)

model = keras.models.Model(x, y)


# In[ ]:

model.summary()


# In[ ]:

X = np.load(path_x)
Y = np.load(path_y)

# print shapes to debug
print(X.shape)
print(Y.shape)


# In[ ]:

optimizer = keras.optimizers.sgd()
loss_wrapper = functools.partial(helper.loss_functions.crazyloss, dim1=64, dim2=64)
loss_wrapper.__name__ = "wrapper"
loss = loss_wrapper
callbacks = [keras.callbacks.TensorBoard(tb_log_dir, histogram_freq=10)]
model.compile(optimizer, loss)


# In[ ]:

model.fit(X, Y, batch_size, epochs, validation_split=0.2, callbacks=callbacks)


# In[ ]:




# In[ ]:




# In[ ]:



