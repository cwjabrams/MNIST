import h5py
import random
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras

from scipy import stats


def load(filename):
	return np.load(filename)

def predict(models, test_data, scaler=None):
	if scaler == None:
		scaler = skp.StandardScaler()
		test_data = scaler.fit_transform(test_data)
	else:
		test_data = scaler.transform(test_data)

	model_predictions = list()

	if len(models) > 1:
		for i in range(len(models)):
			model_predictions.insert(i, models[i].predict(test_data))
	else:
		model_predictions = models[0].predict(test_data)

	predictions = list()

	if len(models) > 1:
		for i in range(len(test_data)):
			lst = list()
			for j in range(len(models)):
				lst.insert(j, np.argmax(model_predictions[j][i]))
			predictions.insert(i, stats.mode(lst)[0][0] + 1)
	else:
		for i in range(len(test_data)):
			predictions.insert(i, np.argmax(model_predictions[i]))

	return predictions

def buildAndTrainModel(training_data, lowerLim, upperLim, labels_at_end=True, scale_relative=False,
	layers=None, learning_rate=.01, decay=1, epochs=2, batch_size=50, print_results=True):

	np.random.shuffle(training_data)

	# Format Data #################################################
	training_set, training_labels, validation_set, validation_labels = getTrainingValidationSets(training_data, labels_at_end=labels_at_end)
	# Format Data #################################################

	in_class_value = lowerLim
	out_of_class_value = upperLim

	# Scale the data ##############################################
	scaler = skp.StandardScaler()
	training_data = scaler.fit_transform(training_set)
	if scale_relative:
		validation_set = scaler.transform(validation_set)
	else:
		validation_set = scaler.fit_transform(validation_set)
	# Scale the data ##############################################


	# Convert Labels to One Hot Encoding ##########################
	tf_training_labels = convertToOneHot(training_labels, lowerLim, upperLim)
	tf_validation_labels = convertToOneHot(validation_labels, lowerLim, upperLim) 
	# Convert Labels to One Hot Encoding ##########################


	# Build Model #################################################
	model = keras.Sequential()
	if layers == None:
		model.add(keras.layers.Dense(200, activation='tanh'))
		model.add(keras.layers.Dense(10, activation='sigmoid'))
	else:
		for layerTuple in layers:
			model.add(keras.layers.Dense(layerTuple[0], activation=layerTuple[1]))

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
		print("Test Accuracy: ", test_acc)
		print("Test Loss: ", test_loss)
		print("\n")

		validation_predictions = predict([model], validation_set, scaler=scaler)
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
	for i in range(len(labels)):
		for j in range(len(clean_labels[0,:])):
			if j == labels[i] - 1:
				clean_labels[i, j] = upperLimit
			else:
				clean_labels[i, j] = lowerLimit 
	return clean_labels

def getTrainingValidationSets(training_data, training_set_factor=0.8, labels_at_end=True):
	training_set_size = round(len(training_data)*.8) 
	if (labels_at_end):
		training_set = training_data[:training_set_size, :-1]
		training_labels = training_data[:training_set_size, -1]

		validation_set = training_data[training_set_size:, :-1]
		validation_labels = training_data[training_set_size:, -1]
	else:
		training_set = training_data[:training_set_size, 1:]
		training_labels = training_data[:training_set_size, 0]

		validation_set = training_data[training_set_size:, 1:]
		validation_labels = training_data[training_set_size:, 0]
	return training_set, training_labels, validation_set, validation_labels


def saveToCSV(filename, predictions):
	f = open(filename, 'w')
	header = 'ImageId,Label\n'
	f.write(header)

	for i in range(len(test_vectors)):
		predicted_class = predictions[i]
		S = str(i + 1) + ',' + str(predicted_class) + '\n'
		f.write(S)
	f.close()