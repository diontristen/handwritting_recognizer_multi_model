# Deep Learning Handwriting Recognition

A simple python application that recognizes handwriting and converts it into text, using a multiple machine models (5). [EMNIST Dataset](https://www.kaggle.com/crawford/emnist) on Kaggle balance data set was used to train the ML. 
This can recognize digits, small and capital letters.

The models were trained on the following characters: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt`

## Things to do before running:
```commandline
mkdir dataset
mkdir generated_pickle
mkdir logs
```

## How to Run:
Train the dataset into a proper model
```commandline
Provide the train.csv, test.csv and mapping.txt.
Also check the train and test data if the orientation is correct, otherwise make necessary adjustments to rotate or flip or even convert it to black on white or vise versa.

RUN:
python3 train_dataset.py
```


Permutate the Model
I used 5 models to predict the input and run a voting system to get the final prediction.

model_1 and model_2:
ARCH: 2 convolutional layers, 2 dense layers, 128 neurons, 0.2 dropout, ran at 10 epochs.

model_3, model_4, and model_5:
ARCH: 2 convolutional layers, 2 dense layers, 64 neurons, 0.2 dropout, ran at 10 epochs.

```commandline
Change the dense_layers, layer_size and conv_layers to desired architecture

RUN:
python3 model_permutation.py
```




PREDICT images
Add additional or completly change images to predict in test/{number}.png
Make sure to use 28 x 28 images.
```
python3 predict.py
```