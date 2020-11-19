# USAGE
# python simple_neural_network.py --dataset kaggle_dogs_vs_cats --model output/simple_neural_network.hdf5

# import the necessary packages
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

dataset_loc = "train_data/train"
new_model = "output/6klasy_gen0_nowy_DS_v4.hdf5"

def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to output model file")
# # args = vars(ap.parse_args())
# args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(dataset_loc))

# initialize the data matrix and labels list
data = []
labels = []

datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    # print(imagePath)

    # img = load_img(imagePath)
    # x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # #
    # # # the .flow() command below generates batches of randomly transformed images
    # # # and saves the results to the `preview/` directory
    # q = 0
    # for batch in datagen.flow(x, batch_size=1, save_prefix=label, save_format='jpeg'):
    #     q += 1
    #     image = cv2.cvtColor(batch[0], cv2.COLOR_RGB2BGR)
    #     #cv2.imshow("Image", image)
    #     cv2.waitKey(0)
    #     features = image_to_feature_vector(image)
    #     data.append(features)
    #     labels.append(label)
    #     if q > 2:
    #         break  # otherwise the generator would loop indefinitely

    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)

    # show an update every 1,000 images ##imagePaths ## dorobić, lepiej
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 6)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

# define the architecture of the network

model = Sequential()
model.add(Dense(1024, input_dim=3072, init="uniform", activation="relu"))
model.add(Dense(500, activation="relu", kernel_initializer="uniform"))
model.add(Dense(6))
model.add(Activation("softmax"))

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(32,32)),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(7, activation="softmax")
# ])


# train the model using SGD
print("[INFO] compiling model...")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=70, batch_size=100, verbose=1)


# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
                                  batch_size=100, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                                                     accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(new_model)
