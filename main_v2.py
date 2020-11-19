from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import argparse
import cv2
import os

datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
dataset_loc = "FF_2.0/dane_testowe_2/train"
# new_model = "output/10klasy_gen0v1.hdf5"

imagePaths = list(paths.list_images(dataset_loc))

for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    img = load_img(imagePath)
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    q = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='FF_2.0/dane_treninigowe_debug/more', save_prefix=label, save_format='jpg'):
        q += 1
#     image = cv2.cvtColor(batch[0], cv2.COLOR_RGB2BGR)
#     #cv2.imshow("Image", image)
#     cv2.waitKey(0)
#     features = image_to_feature_vector(image)
#     data.append(features)
#     labels.append(label)
        if q > 20:
            break  # otherwise the generator would loop indefinitely

