#!/usr/bin/env python
# coding: utf-8

# In[73]:


from __future__ import print_function, division
import tensorflow as tf 
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, MaxPool2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.utils.vis_utils import plot_model
import keras.backend as K
from keras.optimizers import RMSprop
from IPython.display import clear_output
from keras.models import model_from_json, load_model


# In[2]:


pathModelSave = '../NN_training/CEGAN_lines/model_save/'
D_loss_plot = np.array([])
G_loss_plot = np.array([])

dataSetSize = 7600
testSetSize = 500

y2_train = np.empty((dataSetSize,24,144))
y2_train_infos = np.empty((dataSetSize,12), dtype="O")
y2_train_base = np.empty((dataSetSize,24,144))
y2_test = np.empty((testSetSize,24,144))
y2_test_infos = np.empty((testSetSize,12), dtype="O")
y2_test_base = np.empty((testSetSize,24,144))
dataSet = []

for i in range(dataSetSize):
    dataSet.append(np.load('../TrainingDataset/America/Dataset/y2_{}.npy'.format(i), allow_pickle=True, encoding="latin1"))
dataSet = np.asarray(dataSet)


for i in range(dataSetSize-testSetSize):
    y2_train[i] = dataSet[i][1]
    y2_train_base[i] = dataSet[i][0]
    y2_train_infos[i] = dataSet[i][2]
for i in range(dataSetSize-testSetSize,dataSetSize):
    y2_test[i-dataSetSize] = dataSet[i][1]
    y2_test_base[i-dataSetSize] = dataSet[i][0]
    y2_test_infos[i-dataSetSize] = dataSet[i][2]
    
nans = np.empty([])
for i in range(y2_train.shape[0]):
    if math.isnan(np.sum(y2_train[i])):
        nans = np.append(nans, i)
y2_train = np.delete(y2_train, nans.astype(int), axis=0)    
y2_train_base = np.delete(y2_train_base, nans.astype(int), axis=0)
y2_train_infos = np.delete(y2_train_infos, nans.astype(int), axis=0)
scaler = MinMaxScaler(feature_range=(-1,1), copy=True)
for i in range(y2_train.shape[0]):
    array = y2_train[i].flatten()
    array = scaler.fit_transform(array.reshape(-1, 1))
    y2_train[i] = array.reshape(24,144)

y2_train = y2_train.reshape(y2_train.shape[0],24,144,1)
y2_test = y2_test.reshape(y2_test.shape[0],24,144,1)


# In[90]:


maskAsia = []
datasetAsia = []
datasetAsiaInfos = []
testLine = 10

for a in range(5,11):
    datasetAsia.append(np.load('../TrainingDataset/Asia/x_train/y2_{}.npy'.format(a), allow_pickle=True, encoding='latin1'))
    datasetAsiaInfos.append(datasetAsia[a-5][2])
    datasetAsia[a-5] = np.asarray(datasetAsia[a-5][0])
    
datasetAsia = np.asarray(datasetAsia)
datasetAsiaInfos = np.asarray(datasetAsiaInfos)


for i in range(datasetAsia[5].shape[0]):
    if i == testLine:
        maskAsia.append(i)
    if math.isnan(np.sum(datasetAsia[1][i])):
        maskAsia.append(i)

for i in range(datasetAsia.shape[0]):
    arrayAsia = datasetAsia[i].flatten()
    arrayAsia = scaler.fit_transform(arrayAsia.reshape(-1, 1))
    datasetAsia[i] = arrayAsia.reshape(24,144)
    
print('Training mask : {}'.format(maskAsia))


# In[25]:


def build_generator():
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.5))

    # Decoder
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation('tanh'))

#     model.summary()

    masked_img = Input(shape=img_shape)
    gen_missing = model(masked_img)
#     plot_model(model, to_file='model_plot.png', show_shapes=True)
    return Model(masked_img, gen_missing)


# In[26]:


def build_discriminator():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=missing_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
#     model.summary()

    img = Input(shape=missing_shape)
    validity = model(img)
#     plot_model(model, to_file='model_plot.png', show_shapes=True)
    return Model(img, validity)


# In[27]:


def binary_crossentropy_label_smoothing(y_true, y_pred):
    return tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.3)


# In[28]:


img_rows = 24
img_cols = 144
mask_height = 24
mask_width = 144
channels = 1
num_classes = 2
img_shape = (img_rows, img_cols, channels)
missing_shape = (mask_height, mask_width, channels)

optimizer = Adam(0.002, .5)

# Build and compile the discriminator
discriminator = build_discriminator()

# # For the combined model we will only train the generator
discriminator.trainable = False
# 'binary_crossentropy'
discriminator.compile(loss=binary_crossentropy_label_smoothing,
    optimizer=optimizer,
    metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise as input and generates the missing
# part of the image
masked_img = Input(shape=img_shape)
gen_missing = generator(masked_img)

# # The discriminator takes generated images as input and determines
# # if it is generated or if it is a real image
valid = discriminator(gen_missing)

# # # The combined model  (stacked generator and discriminator)
# # # Trains generator to fool discriminator
combined = Model(masked_img , [gen_missing, valid])
combined.compile(loss=['mse', binary_crossentropy_label_smoothing],
    loss_weights=[0.999, 0.001],
    optimizer=optimizer)


# In[29]:


def mask_asia(imgs, mask):
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2])
    masked_imgs = np.empty_like(imgs)
    missing_parts = np.full_like(imgs, 0)
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        for line in mask:
            missing_parts[i][line] = img[line].copy()
            masked_img[line] = np.full(img.shape[1], 0)
        masked_imgs[i] = masked_img
    return np.expand_dims(masked_imgs,3), np.expand_dims(missing_parts,3)


# In[30]:


X_train = y2_train.copy()

def train(epochs, batch_size=128, sample_interval=50, label_switching=False):
    global D_loss_plot, G_loss_plot
    D_loss_plot = np.array([])
    G_loss_plot = np.array([])
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs+1):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        masked_imgs, missing_parts = mask_asia(imgs, maskAsia)
        if not math.isnan(np.sum(imgs)) and not math.isnan(np.sum(masked_imgs)):
            # Generate a batch of new images
            gen_missing = generator.predict(masked_imgs)

            d_loss_real = discriminator.train_on_batch(missing_parts, valid)
            d_loss_fake = discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            clear_output(wait=True)
            D_loss_plot = np.append(D_loss_plot, d_loss[0])
            G_loss_plot = np.append(G_loss_plot, g_loss[0])
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            if epoch % sample_interval == 0:
#                 idx = np.arange(10)
#                 imgs = X_train[idx]
                imgs = datasetAsia
                sample_images(epoch, imgs, datasetAsiaInfos, generator)
        else:
            print("nan value in training set")


# In[153]:


def sample_images(epoch, imgs, imgInfos, model):
    r, c = 3, 6
    finalArray = []
    masked_imgs, missing_parts = mask_asia(imgs, maskAsia)
    gen_missing = model.predict(masked_imgs)

    
    for i in range(c):
        infosArray = imgInfos[i].copy()
        infosArray = np.resize(infosArray, (1,13))
        infosArray = np.append(infosArray, mean_absolute_error(imgs[i][testLine], gen_missing[i][testLine])*100)
        filled_in = imgs[i].copy()
        for line in maskAsia:
            filled_in[line] = np.squeeze(gen_missing[i][line].copy(),1)
        finalArray.append(np.array([imgs[i].reshape(24,144), filled_in.reshape(24, 144), infosArray], dtype='O'))
    np.save('../NN_training/CEGAN_lines/CEGAN_Asia_epoch{}'.format(epoch), np.asarray(finalArray))


# In[154]:


def Save_models():
    generator_json = generator.to_json()
    discriminator_json = discriminator.to_json()

    with open("{}generator_json.json".format(pathModelSave), "w") as json_file:
        json_file.write(generator_json)
    with open("{}discriminator_json.json".format(pathModelSave), "w") as json_file:
        json_file.write(discriminator_json)

    generator.save_weights("{}generator_weights.h5".format(pathModelSave))
    discriminator.save_weights("{}discriminator_weights.h5".format(pathModelSave))


# In[155]:


train(epochs=10, batch_size=64, sample_interval=10, label_switching=False)
Save_models()
np.save('../NN_training/CEGAN_lines/model_save/G_D_losses',np.array([G_loss_plot, D_loss_plot]))


# In[15]:


def Load_CEGAN(path):
    generator_json_load = open("{}generator_json.json".format(path), 'r')
    generator_model_json_read = generator_json_load.read()
    generator_json_load.close()
    generator_model = model_from_json(generator_model_json_read)
    generator_model.load_weights("{}generator_weights.h5".format(path))
    generator_model.save('generator_model.hdf5')
    generator_model=load_model('generator_model.hdf5', compile=True)
    
    discriminator_json_load = open("{}discriminator_json.json".format(path), 'r')
    discriminator_model_json_read = discriminator_json_load.read()
    discriminator_json_load.close()
    discriminator_model = model_from_json(discriminator_model_json_read)
    discriminator_model.load_weights("{}discriminator_weights.h5".format(path))
    discriminator_model.save('discriminator_model.hdf5')
    discriminator_model=load_model('discriminator_model.hdf5', compile=True)
    
    return generator_model, discriminator_model


# In[17]:


gen, dis = Load_CEGAN(pathModelSave)
imgs = datasetAsia
sample_images(0, imgs, datasetAsiaInfos, gen)


# In[19]:


plt.plot(np.load('../../Bureau/IRAP_mount/NN_training/CEGAN_lines/model_save/G_D_losses.npy')[0])
plt.show()
plt.plot(np.load('../../Bureau/IRAP_mount/NN_training/CEGAN_lines/model_save/G_D_losses.npy')[1])
plt.show()

