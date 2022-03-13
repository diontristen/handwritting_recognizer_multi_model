import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import pickle
import time

HEIGHT = 28
WIDTH = 28
class_num = 47

train_x = pickle.load(open("../generated_pickle/train_x.pickle", "rb"))
train_y = pickle.load(open("../generated_pickle/train_y.pickle", "rb"))

test_x = pickle.load(open("../generated_pickle/test_x.pickle", "rb"))
test_y = pickle.load(open("../generated_pickle/test_y.pickle", "rb"))

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=7)

# 1st run: create the following permutations of models
dense_layers = [2]
layer_sizes = [64]
conv_layers = [2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            OPERATION_NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"

            tensorboard = TensorBoard(log_dir=f"logs/{OPERATION_NAME}")

            print(OPERATION_NAME)

            model = Sequential()

            # Begin with Conv2D Layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=(HEIGHT, WIDTH, 1)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # For Extra Conv2D Layers
            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # Flatten the Model to have a single stream
            model.add(Flatten())

            # For Dense Layer
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            # Output Layer
            model.add(Dense(units=class_num, activation='softmax'))

            # Train Model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            model.summary()

            model.fit(
                train_x,
                train_y,
                epochs=10,
                batch_size=32,
                validation_data=(val_x, val_y),
                callbacks=[tensorboard]
            )

            model.save('models/model_5.h5')
