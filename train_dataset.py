import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# defining the directories
DATA_TRAIN_DIR = 'dataset/emnist-balanced-train.csv'  # directory of the dataset used to TRAIN the model
DATA_TEST_DIR = 'dataset/emnist-balanced-test.csv'  # directory of the dataset used to TEST the model
DATA_MAP_DIR = 'dataset/emnist-balanced-mapping.txt'  # directory of the dataset used for the label mapping
GENERATED_PICKLE_DIR = 'generated_pickle'

# defining class related variables
class_mapping = []
class_length = 0

# defining image size
HEIGHT = 28
WIDTH = 28

# defining the train and test dataset with header none
train_pd = pd.read_csv(DATA_TRAIN_DIR, header=None)
test_pd = pd.read_csv(DATA_TEST_DIR, header=None)
map_pd = pd.read_csv(DATA_MAP_DIR, delimiter=' ', header=None, usecols=[1]).squeeze(True)

# remapping the map labels to local variable class_mapping
for num in range(len(map_pd)):
    class_mapping.append(chr(map_pd[num]))

# determining the length of class map
class_length = len(class_mapping)


# generate an image base on the test number
def covert_training_data(dataset, image_number):
    test_row_num_img = dataset.values[image_number, 1:]
    reshape_img = test_row_num_img.reshape(HEIGHT, WIDTH)
    reshape_img = np.transpose(reshape_img, axes=[1, 0])
    return reshape_img


# return the label of the row
def get_char(dataset, row):
    return class_mapping[dataset.values[row, 0]]


# return the generated dataset with proper sets
def generate_proper_datasets(dataset):
    result = []
    for i in range(len(dataset)):
        result.append(covert_training_data(dataset, i))
    return np.asarray(result)


# return the normalized dataset
def normalize_dataset(dataset):
    result = dataset.astype('float32')
    return result / 255


# generate the proper dataset
train_x = generate_proper_datasets(train_pd)
test_x = generate_proper_datasets(test_pd)

# normalize the train and test data
train_x = normalize_dataset(train_x)
test_x = normalize_dataset(test_x)

# creating the answers
train_y = train_pd.iloc[:, 0]
test_y = test_pd.iloc[:, 0]

# for later
train_y_1D = train_y
test_y_1D = test_y

train_x_3D = train_x
test_x_3D = test_x


# prepare datasets for ML algorithm
train_y = tf.keras.utils.to_categorical(train_y, class_length)
test_y = tf.keras.utils.to_categorical(test_y, class_length)


# generate pickle for the dataset
def generate_pickle(title, dataset):
    print(f'generating {title}.pickle')
    pickle_out = open(f"{GENERATED_PICKLE_DIR}/{title}.pickle", 'wb')
    pickle.dump(dataset, pickle_out)
    pickle_out.close()


generate_pickle('train_x', train_x)
generate_pickle('test_x', test_x)
generate_pickle('train_y', train_y)
generate_pickle('test_y', test_y)



