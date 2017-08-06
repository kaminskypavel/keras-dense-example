from sklearn.model_selection import train_test_split

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

# encode training set
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
Y_hat_train = np_utils.to_categorical(encoded_Y)

# encode testing set
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y = encoder.transform(Y_test)
Y_hat_test = np_utils.to_categorical(encoded_Y)

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

model.fit(X_train, Y_hat_train, epochs=EPOCHS, batch_size=5, verbose=DEBUG_LEVEL, callbacks=[tensorBoardCallBack],
          validation_data=(X_test, Y_hat_test))
