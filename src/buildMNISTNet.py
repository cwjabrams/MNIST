import h5py
import random
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras

from scipy import stats


###############################################################################
# Data Prep
##############################################################################

training_complete = np.load("../bin/training_data.npy")

np.random.shuffle(training_complete)

training_set_size = round(len(training_complete)*.8) 

training_data = training_complete[:training_set_size, 1:]
training_labels = training_complete[:training_set_size, 0]

validation_data = training_complete[training_set_size:, 1:]
validation_labels = training_complete[training_set_size:, 0]

in_class_value = .88
out_of_class_value = .12

############################################################################
# Set Labels to be sets of one-hot encoded vectors dimension 26
############################################################################

def convertToOneHot(labels, lowerLimit, upperLimit):
	clean_labels = np.zeros((len(labels), len(set(labels))))
	for i in range(len(training_labels)):
		for j in range(len(clean_labels[0,:])):
			if j == training_labels[i] - 1:
				clean_labels[i, j] = lowerLimit
			else:
				clean_labels[i, j] = upperLimit 
	return clean_labels


clean_labels = np.zeros((len(training_labels), 10))
for i in range(len(training_labels)):
	for j in range(len(clean_labels[0,:])):
		if j == training_labels[i] - 1:
			clean_labels[i, j] = in_class_value
		else:
			clean_labels[i, j] = out_of_class_value

val_labels = np.zeros((len(validation_labels), 10))
for i in range(len(validation_labels)):
	for j in range(len(val_labels[0,:])):
		if j == validation_labels[i] - 1:
			val_labels[i, j] = in_class_value
		else:
			val_labels[i, j] = out_of_class_value

# Save a random image
image = np.reshape(training_data[0], (28,28))
plt.figure()
plt.imshow(image)
plt.colorbar() 
plt.gca().grid(False)
plt.savefig("../images/random_letter")

############################################################################

scaler = skp.StandardScaler()
training_data = scaler.fit_transform(training_data)
validation_data = scaler.transform(validation_data)

################################################################################
# Neural Net Training
################################################################################

models = list()

for i in range(100):
	model = keras.Sequential()
	if i < 50:
		model.add(keras.layers.Dense(800, activation='relu'))
		model.add(keras.layers.Dense(10, activation='softmax'))
	else:
		model.add(keras.layers.Dense(400, activation='tanh'))
		model.add(keras.layers.Dense(10, activation='sigmoid'))

	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.2
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           len(training_data), 0.90, staircase=True)

	model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate), loss='categorical_crossentropy',
					metrics=[keras.metrics.categorical_accuracy])

	data_labels = list(zip(training_data, clean_labels))
	random.shuffle(data_labels)
	training_data, clean_labels = zip(*data_labels)
	training_data, clean_labels = np.array(training_data), np.array(clean_labels)

	if i < 25:
		model.fit(training_data, clean_labels, epochs=10, batch_size=100)
	elif i < 50:
		model.fit(training_data, clean_labels, epochs=10, batch_size=75)
	elif i < 75:
		model.fit(training_data, clean_labels, epochs=10, batch_size=50)
	else:
		model.fit(training_data, clean_labels, epochs=10, batch_size=25)


	test_loss, test_acc = model.evaluate(validation_data, val_labels)
	print("\n")
	print("Test Accuracy", i, ": ", test_acc)
	print("Test Loss", i, ": ", test_loss)
	print("\n")
	models.insert(i, model)


model_validation_predictions = list()
average_validation_predictions = list()

for i in range(len(models)):
	model_validation_predictions.insert(i, models[i].predict(validation_data))


for i in range(len(validation_data)):
	lst = list()
	for j in range(len(models)):
		lst.insert(j, np.argmax(model_validation_predictions[j][i]))
	average_validation_predictions.insert(i, stats.mode(lst)[0][0] + 1)
	

total = 0
for i in range(len(validation_data)):
	if i < 20:
		print("Validation Prediction: ", average_validation_predictions[i], "\nValidation Label:", validation_labels[i])
	if average_validation_predictions[i] == validation_labels[i]:
		total += 1

correct = total / len(validation_labels)
print("Validation Accuracy: ", correct)


test_vectors = np.load("../bin/test_data.npy")
test_vectors = scaler.transform(test_vectors)

model_predictions = list()

for i in range(len(models)):
	model_predictions.insert(i, models[i].predict(test_vectors))

predictions = list()

for i in range(len(test_vectors)):
	lst = list()
	for j in range(len(models)):
		lst.insert(j, np.argmax(model_predictions[j][i]))
	predictions.insert(i, stats.mode(lst)[0][0] + 1)

f = open('kaggle_submission.csv', 'w')
header = 'ImageId,Label\n'
f.write(header)

for i in range(len(test_vectors)):
	predicted_class = predictions[i]
	S = str(i + 1) + ',' + str(predicted_class) + '\n'
	f.write(S)
f.close()




