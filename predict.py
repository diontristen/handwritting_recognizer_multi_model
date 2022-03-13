import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image


model_1 = tf.keras.models.load_model('models/model_1.h5')
model_2 = tf.keras.models.load_model('models/model_2.h5')
model_3 = tf.keras.models.load_model('models/model_3.h5')
model_4 = tf.keras.models.load_model('models/model_4.h5')
model_5 = tf.keras.models.load_model('models/model_5.h5')

DATA_MAP_DIR = 'dataset/emnist-balanced-mapping.txt'  # directory of the dataset used for the label mapping

map_pd = pd.read_csv(DATA_MAP_DIR, delimiter=' ', header=None, usecols=[1]).squeeze(True)

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
class_length = 0

# # remapping the map labels to local variable class_mapping
# for num in range(len(map_pd)):
#     class_mapping.append(chr(map_pd[num]))

# determining the length of class map
class_length = len(class_mapping)

train_x = pickle.load(open("generated_pickle/train_x.pickle", "rb"))
train_y = pickle.load(open("generated_pickle/train_y.pickle", "rb"))

test_x = pickle.load(open("generated_pickle/test_x.pickle", "rb"))
test_y = pickle.load(open("generated_pickle/test_y.pickle", "rb"))

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=7)

def make_prediction(model, img):
    prediction = model.predict(img)
    idx_prediction = np.argmax(prediction[0])
    return class_mapping[idx_prediction]

def model_jury_ruling(*argv):
    all_predictions = []
    for arg in argv:
        all_predictions.append(arg)

    hash = {}

    for prediction in all_predictions:
        if prediction not in hash:
            hash[prediction] = 1
        else:
            hash[prediction] += 1

    return max(hash, key=hash.get)


def take_all_prediction(image):
    predictions = []
    models = [model_1, model_2, model_3, model_4, model_5]
    for model in models:
        predicted = make_prediction(model, image)
        predictions.append(predicted)
    return predictions


def prepare(image_number):
    IMG_SIZE = 28
    img = cv2.imread('test/{}.png'.format(image_number), cv2.IMREAD_GRAYSCALE)
    img = np.invert(np.array([img]))
    array_of_chars = [img]
    char_img_converted_to_in_sample = []
    char_img_heights = []
    char_img_widths = []

    for char_img in array_of_chars:
        # trim off all excess pixels and center the char_img up
        # char_img = center_image(char_img)
        char_img_heights.append(len(char_img))
        char_img_widths.append(len(char_img[0]))

        # we will now begin padding
        # convert to a numpy array
        char_img = np.array(char_img, dtype='float32')

        # resize the image, reshape, make prediction
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))
        char_img /= 255

        # plt.imshow(char_img, cmap=plt.cm.binary)
        # plt.show()

        char_img = char_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        char_img_converted_to_in_sample.append(char_img)
    return char_img_converted_to_in_sample, 1, char_img_heights, char_img_widths


image_number = 1
while os.path.isfile('test/{}.png'.format(image_number)):
    try:
        final_images, space_location, char_img_heights, char_img_widths = prepare(image_number)
        final_prediction = []
        for idx, img in enumerate(final_images):

            char_prediction_1 = make_prediction(model_1, img)
            char_prediction_2 = make_prediction(model_2, img)
            char_prediction_3 = make_prediction(model_3, img)
            char_prediction_4 = make_prediction(model_4, img)
            char_prediction_5 = make_prediction(model_5, img)

            print('\n', char_prediction_1, char_prediction_2, char_prediction_3, char_prediction_4, char_prediction_5)

            # Combined prediction
            final_char_prediction = model_jury_ruling(char_prediction_1, char_prediction_2, char_prediction_3,
                                                      char_prediction_4, char_prediction_5)
            im = Image.open(f"test/{image_number}.png")
            LOWER_CASE = im.height * .50
            tall_uniform_lc_letters = 'klpy'
            TALL_UNIFORM_LC = im.height * .70

            if final_char_prediction.isnumeric() == False and final_char_prediction.lower() not in class_mapping:
                if final_char_prediction.lower() in tall_uniform_lc_letters:
                    if char_img_heights[idx] < TALL_UNIFORM_LC:
                        final_char_prediction = final_char_prediction.lower()
                elif char_img_heights[idx] < LOWER_CASE:
                    final_char_prediction = final_char_prediction.lower()

            # Typically, "zeroes" (0) are fairly large, we would typically rather have "o" instead if a 0 if a user makes them small
            if final_char_prediction == '0' and char_img_heights[idx] < LOWER_CASE:
                final_char_prediction = 'o'


            final_prediction.append(final_char_prediction)
            print('Final Prediction: ', final_prediction)
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f"Prediction: {final_char_prediction}")
            plt.show()



    #
    except:
        print("Error reading image! Proceeding with next image...")
    finally:
        image_number += 1