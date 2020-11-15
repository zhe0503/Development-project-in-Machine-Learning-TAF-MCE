import pandas
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


def ourCNN():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=5e-4), metrics=['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	print(model.summary())
	return model
