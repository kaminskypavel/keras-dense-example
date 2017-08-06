EPOCHS = 100
DEBUG_LEVEL = 1

import keras
import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu', name="layer1"))
model.add(Dense(3, activation='softmax', name="layer2"))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit & log
tensorBoardCallBack = keras.callbacks.TensorBoard(log_dir='./logs',
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_images=True)

model.fit(X, dummy_y, epochs=EPOCHS, batch_size=5, verbose=DEBUG_LEVEL, callbacks=[tensorBoardCallBack],
          validation_split=0.33)

