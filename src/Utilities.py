import h5py
import random
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras

from scipy import stats


def predict(models, test_data):
	return None		

def buildAndTrainModel(training_data, lowerLim, upperLim, labels_at_end=True, scale_relative=False,
	layers=None, learning_rate=.01, decay=1, epochs=2, batch_size=50, print_results=True ):

	np.random.shuffle(training_data)

	# Format Data #################################################
	training_set_size = round(len(training_data)*.8) 
	if (labels_at_end):
		training_set = training_complete[:training_set_size, :-1]
		training_labels = training_complete[:training_set_size, -1]

		validation_set = training_complete[training_set_size:, :-1]
		validation_labels = training_complete[training_set_size:, -1]
	else:
		training_set = training_complete[:training_set_size, 1:]
		training_labels = training_complete[:training_set_size, 0]

		validation_set = training_complete[training_set_size:, 1:]
		validation_labels = training_complete[training_set_size:, 0]

	# Format Data #################################################

	in_class_value = lowerLim
	out_of_class_value = upperLim

	# Scale the data ##############################################
	scaler = skp.StandardScaler()
	training_data = scaler.fit_transform(training_set)
	if scale_relative:
		validation_data = scaler.transform(validation_set)
	else:
		validation_data = scaler.fit_transform(validation_set)
	# Scale the data ##############################################


	# Convert Labels to One Hot Encoding ##########################
	tf_training_labels = convertToOneHot(training_labels, lowerLim, upperLim)
	tf_validation_labels = convertToOneHot(validation_labels) 
	# Convert Labels to One Hot Encoding ##########################


	# Build Model #################################################
	model = keras.Sequential()
	if layers == None:
		model.add(keras.layers.Dense(400, activation='tanh'))
		model.add(keras.layers.Dense(10, activation='sigmoid'))
	else:
		for layerTuple in layers:
			model.add(keras.layers.Dense(layerTuple[0], layerTuple[1]))

	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = learning_rate
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           len(training_set), decay, staircase=True)

	model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate), loss='categorical_crossentropy',
					metrics=[keras.metrics.categorical_accuracy])

	model.fit(training_set, tf_training_labels, epochs=epochs, batch_size=batch_size)
	# Build Model #################################################

	# Check Validation Accuracy ###################################
	if (print_results):
		test_loss, test_acc = model.evaluate(validation_set, tf_validation_labels)
		print("\n")
		print("Test Accuracy", i, ": ", test_acc)
		print("Test Loss", i, ": ", test_loss)
		print("\n")

		validation_predictions = model.predict(validation_set)
		for i in range(len(validation_predictions)):
			validation_predictions[i] = np.argmax(validation_predictions[i])
		total = 0
		for i in range(len(validation_set)):
			if validation_predictions[i] == validation_labels[i]:
				total += 1

		correct = total / len(validation_labels)
		print("Validation Accuracy: ", correct)
	# Check Validation Accuracy ###################################

	return model

def convertToOneHot(labels, lowerLimit=0, upperLimit=1):
	clean_labels = np.zeros((len(labels), len(set(labels))))
	for i in range(len(training_labels)):
		for j in range(len(clean_labels[0,:])):
			if j == training_labels[i] - 1:
				clean_labels[i, j] = lowerLimit
			else:
				clean_labels[i, j] = upperLimit 
	return clean_labels
