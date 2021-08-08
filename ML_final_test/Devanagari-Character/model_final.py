from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D as Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
import cv2
import csv
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

img_rows, img_cols = 256, 256
img_channels = 1

path = 'dataset_original/'

#listim = os.listdir(path + 'final_images/')

X_tr = []
X_ts = []
Y_tr = []
Y_ts = []

for i in xrange(1,1851):
	tp1 = cv2.imread( path + 'final_images/' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
	X_tr.append(cv2.resize(tp1, (64,64), interpolation = cv2.INTER_NEAREST).reshape(64,64,1))


for i in xrange(1851,1967):
	tp1 = cv2.imread( path + 'final_images/' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
	X_ts.append(cv2.resize(tp1, (64,64), interpolation = cv2.INTER_NEAREST).reshape(64,64,1))


with open(path + 'train3.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		tp2 = row[1:]
		#print(type(tp2))
		my_out = np.zeros(128)
		for p, _p in zip(np.array(tp2)=='', tp2):
			if not p:
				my_out[int(_p)] = 1
		Y_tr.append(my_out)

with open(path + 'train4.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		tp2 = row[1:]
		my_out = np.zeros(128)
		for p, _p in zip(np.array(tp2)=='', tp2):
			if not p:
				my_out[int(_p)] = 1
		Y_ts.append(my_out)

X_tr = np.array(X_tr)
Y_tr = np.array(Y_tr)
print(X_tr.shape, Y_tr.shape)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(X_tr.shape[1], X_tr.shape[1], 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


model.fit(X_tr, Y_tr, batch_size=20, epochs=5, verbose=1, validation_split=0.1)
#score = model.evaluate(x_test, y_test, batch_size=32,epochs = 20, verbose=1, validation_split=0.1)

#fname = "weights-Test-CNN.hdf5"
model.save('ocr_model')


"""with open('dataset_original/pred4.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')

    xt = []
    for i in os.listdir('dataset_original/final_images/'):
    	print i
    	x1 = cv2.imread('dataset_original/final_images/' + str(i) ,cv2.IMREAD_GRAYSCALE)
    	x = cv2.resize(x1, (64,64), interpolation = cv2.INTER_NEAREST).reshape(64,64,1)

    	xt.append(x)


	xt = np.array(xt)
	print xt.shape
	y = model.predict(xt)
	spamwriter.writerow([y])"""
