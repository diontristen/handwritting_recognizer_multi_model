import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

TEST_CASE_SMALL = True
TEST_DIR_CAPITAL = 'test/capital'
TEST_DIR_SMALL = 'test/small'

TEST_DIR = TEST_DIR_CAPITAL
if TEST_CASE_SMALL:
    TEST_DIR = TEST_DIR_SMALL

# defining the models
model_1 = tf.keras.models.load_model('models/model_1.h5')  # 2 dense layer 2 convo layer 128 nodes
model_2 = tf.keras.models.load_model('models/model_2.h5')  # 2 dense layer 2 convo layer 128 nodes
model_3 = tf.keras.models.load_model('models/model_3.h5')  # 2 dense layer 2 convo layer 64 nodes
model_4 = tf.keras.models.load_model('models/model_4.h5')  # 2 dense layer 2 convo layer 64 nodes
model_5 = tf.keras.models.load_model('models/model_5.h5')  # 2 dense layer 2 convo layer 64 nodes

# defining mapping
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
class_length = len(class_mapping)

IMG_SIZE = 28

def make_prediction(model, image):
    prediction = model.predict(image)
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


def prepare(test_number):
    image = cv2.imread(f"{TEST_DIR}/{test_number}.png", cv2.IMREAD_GRAYSCALE)
    image = np.invert(np.array([image]))
    array_of_chars = [image]
    char_img_converted_to_in_sample = []
    char_img_heights = []
    char_img_widths = []

    for char_img in array_of_chars:
        char_img_heights.append(len(char_img))
        char_img_widths.append(len(char_img[0]))

        # convert to a numpy array
        char_img = np.array(char_img, dtype='float32')

        # resize the image, reshape, make prediction
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))

        # normalize the image
        char_img /= 255
        char_img = char_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        char_img_converted_to_in_sample.append(char_img)
    return char_img_converted_to_in_sample, 1, char_img_heights, char_img_widths



image_number = 1
while os.path.isfile(f"{TEST_DIR}/{image_number}.png"):
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

            im = Image.open(f"{TEST_DIR}/{image_number}.png")
            LOWER_CASE = im.height * .50
            TALL_UNIFORM_LC = im.height * .70

            tall_uniform_lc_letters = 'klpy'

            if final_char_prediction.isnumeric() == False and final_char_prediction.lower() not in class_mapping and TEST_CASE_SMALL == True:
                if final_char_prediction.lower() in tall_uniform_lc_letters:
                    if char_img_heights[idx] < TALL_UNIFORM_LC:
                        final_char_prediction = final_char_prediction.lower()
                elif char_img_heights[idx] < LOWER_CASE:
                    final_char_prediction = final_char_prediction.lower()

            if final_char_prediction == '0' and char_img_heights[idx] < LOWER_CASE and TEST_CASE_SMALL == True:
                final_char_prediction = 'o'

            final_prediction.append(final_char_prediction)
            print('Final Prediction: ', final_prediction)
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f"Testing Number: {image_number} \n Prediction: {final_char_prediction}")
            plt.show()

    except:
        print("Error reading image! Proceeding with next image...")
    finally:
        image_number += 1
