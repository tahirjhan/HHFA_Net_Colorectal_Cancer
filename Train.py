# Import libraries
import tensorflow.compat.v1 as tf
import os
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input,Lambda,Reshape, TimeDistributed,ConvLSTM2D, Dense,RepeatVector, Flatten,GlobalAveragePooling2D, Dropout, BatchNormalization, Add, multiply
from keras.layers import Conv2D, MaxPool2D,LSTM,Bidirectional, LeakyReLU, Activation, AveragePooling2D,SeparableConv2D
from keras.optimizers import Adam,Nadam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from keras.models import load_model
seed = 232
np.random.seed(seed)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("CUDA working?...")
print(tf.test.is_built_with_cuda())

print("Data directory...")
cfolder = os.getcwd()
print(cfolder)
input_path = cfolder

img_dims = [150,150]
epochs = 50
batch_size = 32

# Read data
def process_data(img_dims, batch_size):
    if len(img_dims) > 1:
        r = img_dims[0];
        c = img_dims[1]
    else:
        r = img_dims, c = img_dims

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        subset='training',
        directory=os.path.join(input_path, 'Training'),
        target_size=(r, c),
        batch_size=batch_size,
        class_mode='categorical',
        seed=123,
        shuffle=True)

    val_gen = train_datagen.flow_from_directory(
        subset='validation',
        directory=os.path.join(input_path,'Training'),
        target_size=(r, c),
        batch_size=batch_size,
        class_mode='categorical',
        seed=123,
        shuffle=True)
    return train_gen, val_gen

# Read data
train_gen,val_gen = process_data(img_dims, batch_size)

# Check class indices
class_dict = train_gen.class_indices
print(class_dict)

#----------------------------------
# Dynamic learning rate
# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Crop and Concat function------------------------------
def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        return keras.layers.Concatenate(axis=3)([x1, x2])

# Single convolutional block in proposed network
def conv_block(inputs, num_filters, pool):
    x1 = SeparableConv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x2 = SeparableConv2D(filters=num_filters, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
    x = Add()([x1, x2])
    #    x = crop_and_concat(x1,x2)
    #    x = SeparableConv2D(filters=(num_filters), kernel_size=(1,1), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, trainable=True)(x)
    # x = BatchNormalization() (x)
    if pool:
        x = AveragePooling2D(pool_size=(2, 2))(x)
    return x

# HHFA-Net
r = 150;
c = 150;
inputs = Input(shape=(r, c, 3))

x1 = conv_block(inputs, 16, 0)
x2 = conv_block(x1, 32, 0)
c1 = crop_and_concat(x1, x2)
pool_1 = AveragePooling2D(pool_size=(2, 2))(x2)

x3 = conv_block(pool_1, 64, 0)
x4 = conv_block(x3, 128, 0)
c2 = crop_and_concat(x3, x4)
pool_2 = AveragePooling2D(pool_size=(2, 2))(x4)

c1 = AveragePooling2D(pool_size=(2, 2))(c1)
c1_c2 = crop_and_concat(c1, c2)

x5 = conv_block(pool_2, 128, 0)
x6 = conv_block(x5, 256, 0)
c3 = crop_and_concat(x5, x6)
pool_3 = AveragePooling2D(pool_size=(2, 2))(x6)

x7 = conv_block(pool_3, 256, 0)
x8 = conv_block(x7, 512, 0)
c4 = crop_and_concat(x7, x8)
pool_4 = AveragePooling2D(pool_size=(2, 2))(x8)
# c3_c4 = crop_and_concat(pool_3,c4)
x9 = conv_block(pool_4, 1024, 0)

ad1 = AveragePooling2D(pool_size=(4, 4))(c1_c2)
ad2 = AveragePooling2D(pool_size=(2, 2))(ad1)
cnn_features = crop_and_concat(x9, ad2)
print(cnn_features.shape)

x = GlobalAveragePooling2D()(cnn_features)
x = Dropout(rate=0.5)(x)
output = Dense(units=7, activation='softmax')(x)
model = Model(inputs=inputs, outputs=output)

# Summary
model.summary()

# Define training parameters
adamc = Adam(learning_rate=0.001)
model.compile(optimizer= adamc, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='HHFA_Net_bestweights.hdf5', save_best_only=True, save_weights_only=True)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
Early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Training model
hist = model.fit_generator(train_gen,steps_per_epoch=train_gen.samples //
                           batch_size,epochs=epochs, validation_data=val_gen,validation_steps=val_gen.samples //
                           batch_size, callbacks=[checkpoint, reduce_lr, Early_stop])
# Save history for training, validation curves
np.save('my_history.npy',hist.history)

#Training and validation accuracy and loss curves
train_loss = hist.history['loss']
train_acc  = hist.history['accuracy']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_ylabel('Loss')
ax1.set_xlabel('Epochs')
ax1.plot(hist.history['loss'], color='Red',linewidth = '2')
ax1.axis([0,50,0.0,1.0])

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.plot(hist.history['accuracy'], color='DarkBlue', linewidth = '2')
ax2.axis([0,50,0.7,1.0])
plt.show()







