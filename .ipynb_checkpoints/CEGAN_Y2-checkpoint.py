from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import math

numberOfFiles=0
try: 
    filesXtrain = os.listdir('../TrainingDataset/x_train/'); 
    numberOfFiles = len(filesXtrain)
except: print('File not found')
numberOfFiles-=11
testingSetSize = 100
y2_train = np.empty((numberOfFiles-testingSetSize,24,144))
y2_test = np.empty((testingSetSize,24,144))

for i in range(0,numberOfFiles-testingSetSize):
    y2_train[i] = np.load('../TrainingDataset/x_train/Y2_36_60_{}.npy'.format(i))
for i in range(numberOfFiles-testingSetSize,numberOfFiles):
    y2_test[i-(numberOfFiles-testingSetSize)-1] = np.load('../TrainingDataset/x_train/Y2_36_60_{}.npy'.format(i))

nans = np.empty([])
for i in range(y2_train.shape[0]):
    if math.isnan(np.sum(y2_train[i])):
        nans = np.append(nans, i)
y2_train = np.delete(y2_train, nans, axis=0)    

scaler = MinMaxScaler(feature_range=(-1,1), copy=True)
for i in range(y2_train.shape[0]):
    array = y2_train[i].flatten()
    array = scaler.fit_transform(array.reshape(-1, 1))
    y2_train[i] = array.reshape(24,144)

y2_train = y2_train.reshape((numberOfFiles-testingSetSize)-nans.shape[0],24,144,1)
y2_test = y2_test.reshape(testingSetSize,24,144,1)


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
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation('tanh'))

    masked_img = Input(shape=img_shape)
    gen_missing = model(masked_img)
    return Model(masked_img, gen_missing)


def build_discriminator():

	model = Sequential()

	model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=missing_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(256, kernel_size=3, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	img = Input(shape=missing_shape)
	validity = model(img)

	return Model(img, validity)

img_rows = 24
img_cols = 144
mask_height = 12
mask_width = 72
channels = 1
num_classes = 2
img_shape = (img_rows, img_cols, channels)
missing_shape = (mask_height, mask_width, channels)

optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()

# # For the combined model we will only train the generator
discriminator.trainable = False

discriminator.compile(loss='binary_crossentropy',
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
combined.compile(loss=['mse', 'binary_crossentropy'],
    loss_weights=[0.999, 0.001],
    optimizer=optimizer)

def mask_randomly(imgs):
    y1 = np.random.randint(0, img_rows - mask_height, imgs.shape[0])
    y2 = y1 + mask_height
    x1 = np.random.randint(0, img_cols - mask_width, imgs.shape[0])
    x2 = x1 + mask_width

    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((imgs.shape[0], mask_height, mask_width, channels))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
        masked_img[_y1:_y2, _x1:_x2, :] = 0
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)



X_train = y2_train.copy()
def train(epochs, batch_size=128, sample_interval=50):

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        sys.stdout.flush()
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        masked_imgs, missing_parts, _ = mask_randomly(imgs)
        if not math.isnan(np.sum(imgs)) and not math.isnan(np.sum(masked_imgs)):
            # Generate a batch of new images
            gen_missing = generator.predict(masked_imgs)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(missing_parts, valid)
            d_loss_fake = discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            print("Epoch {} out of {}".format(epoch, epochs))
            if epoch % sample_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                sample_images(epoch, imgs)
        else:
            print("nan value in training set")

def sample_images(epoch, imgs):
    r, c = 3, 6

    masked_imgs, missing_parts, (y1, y2, x1, x2) = mask_randomly(imgs)
    gen_missing = generator.predict(masked_imgs)

    imgs = 0.5 * imgs + 0.5
    masked_imgs = 0.5 * masked_imgs + 0.5
    gen_missing = 0.5 * gen_missing + 0.5

    for i in range(c):
        np.save("matrices_CEGAN/base_{}_{}".format(epoch, i), imgs[i, :,:].reshape(24,144))
        np.save("matrices_CEGAN/masked_{}_{}".format(epoch, i), masked_imgs[i, :,:].reshape(24, 144))
        filled_in = imgs[i].copy()
        filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
        np.save("matrices_CEGAN/filled_{}_{}".format(epoch, i), filled_in.reshape(24, 144))


def save_model():

    def save(model, model_name):
        model_path = "saved_model_CEGAN/%s.json" % model_name
        weights_path = "saved_model_CEGAN/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "generator")
    save(discriminator, "discriminator")


train(epochs=5000, batch_size=128, sample_interval=100)
save_model()
