'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
test
'''

from __future__ import print_function
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K, preprocessing as P
import matplotlib.pyplot as plt

from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# dimensions of our images.
img_width, img_height = 75, 75

train_data_dir = 'data_3/train'
validation_data_dir = 'data_3/validation'
nb_train_samples = 400
nb_validation_samples = 130
epochs = 256
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Creating a Sequential model
model = Sequential()
model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='tanh', input_shape=(img_width, img_height, 3,)))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',  # adam, rmsprop
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save_weights('first_try.h5')


# load the network
# model = load_model(args["model"])

def image_to_feature_vector(image_local, size=(100, 100)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image_local, size).flatten()


CLASSES = ['Cereal', 'Chips', 'Chocolate', 'Deo', 'Ketchup', 'Milk', 'Shampoo']


# loop over our testing images
for imagePath in paths.list_images('data_3/validation/Milk'):

    print("[INFO] classifying {}".format(
        imagePath[imagePath.rfind("/") + 1:]))
    image = P.image.load_img(imagePath, target_size=(img_height, img_width))
    x = P.image.img_to_array(image)
    x = x.reshape(1, img_width, img_height, 3).astype('float')
    x /= 255

    probabilities = model.predict(x)
    prediction = probabilities.argmax(axis=0)

    label = "prediction: " + CLASSES[prediction] + " " + probabilities[prediction]*100 + "%"

    cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

