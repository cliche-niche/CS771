from params import *
import numpy as np
import utils
import tensorflow as tf


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

(X, y_) = utils.loadData( "../../train", dictSize = dictSize )
y = y_.astype(int)
y = np.reshape(y,(y.shape[0],1))
# y = np.zeros((y_.shape[0], CLASSES + 1))
# y[np.arange(y_.size), y_] = 1
print(y.shape)
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(dictSize,)))
model.add(tf.keras.layers.Dense(dictSize, activation='relu'))
model.add(tf.keras.layers.Dense(CLASSES+1, activation = 'softmax'))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.05,
    decay_steps=100,
    decay_rate=0.95)

optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule)  # Optimizer

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
loss_value = 50
for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for i in range(0,(int)(TRAIN_SET_SIZE/BATCH_SIZE)):
        x_batch_train = X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        x_batch_train = x_batch_train.toarray()
        # Divide by norm
        x_batch_train = x_batch_train / np.linalg.norm(x_batch_train, axis=1, keepdims=True)
        x_batch_train = tf.convert_to_tensor(x_batch_train)
        y_batch = tf.convert_to_tensor(y[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            
            preds = model(x_batch_train, training=True)  # Logits for this minibatch
          
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch, preds)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
    print(epoch,tf.math.reduce_mean(loss_value))
model.save("models/dl.zip")
# print(model(X).shape)
# history = model.fit(
#     X,
#     y,
#     batch_size=64,
#     epochs=2,
# )