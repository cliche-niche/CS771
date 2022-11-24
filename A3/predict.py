# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR
import tensorflow as tf
import numpy as np
import pickle as pkl
import cv2 as cv
from utils import *
# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.
model = tf.keras.models.load_model("model.h5")
index_to_label = pkl.load(open("index_to_label.pkl", "rb"))
def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration
	codes = []
	for file in filenames:
		img = cv.imread(file)
		images = process(img)
		if(len(images)!=3):
			img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			images = [img[0:140, 13:153], img[9:140+9, 200:200+140], img[0:140, 340:340+140]]
		code = []
		for i in range(len(images)):
			im = cv.resize(images[i], (64,64))
			im = im/255
			index = model.predict(im.reshape(1, 64, 64, 1), verbose = False)
			code.append(index_to_label[np.argmax(index)])
		codes.append(",".join(code))
	
	return codes