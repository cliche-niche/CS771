from sklearn import *
import os
import cv2 as cv
import tensorflow as tf
import numpy as np
import pickle as pkl
from scipy import ndimage
from utils import *
# Generate data
# Read outs directory
outs_dir = "outs"
dataset, labels = generate_training_data()
index_to_label = pkl.load(open("index_to_label.pkl", "rb"))
label_to_index = pkl.load(open("label_to_index.pkl", "rb"))
# Apply dictinary to labels
labels_ = [label_to_index[label] for label in labels]
# Convert to numpy array
dataset = np.array(dataset)
labels_ = np.array(labels_)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax')
])

# Randomly check labels


# Train model
# Compile model
# Set high learning rate , sparse_categorical_crossentropy, and accuracy metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
# Train model
model.fit(dataset, labels_, epochs=100, batch_size=256)
# Save model
model.save("model.h5")



