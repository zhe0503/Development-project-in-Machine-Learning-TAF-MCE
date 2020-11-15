import pandas
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def ourDNN(inputdims):
	# create model
	model = Sequential()
	model.add(Dense(inputdims, input_dim=inputdims, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=5e-4), metrics=['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
	print(model.summary())
	return model
	
	
	

def draw(history):

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	plt.figure(figsize=(8, 8))
	plt.subplot(2, 1, 1)
	plt.plot(acc, label='Training Accuracy')
	plt.plot(val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.ylabel('Accuracy')
	plt.ylim([min(plt.ylim()),1])
	plt.title('Training and Validation Accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.ylabel('Cross Entropy')
	plt.ylim([0,1.0])
	plt.title('Training and Validation Loss')
	plt.xlabel('epoch')
	plt.show()
