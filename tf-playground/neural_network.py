import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

m, n = housing.data.shape

def scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

scaled_data =  scale(housing.data)
housing_with_bias = np.c_[np.ones((m, 1)), scaled_data]

n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def get_batch(X, y, batch_size):
    indicies = np.random.choice(len(X), size=batch_size)
    return X[indicies], y[indicies]

def layer(X, n_inputs, n_outputs):
    with tf.name_scope('layer'):
        w = tf.Variable(tf.random_normal((n_inputs, n_outputs)), name='weights')
        b = tf.zeros((n_outputs), name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, 0, name='relu')

X = tf.placeholder(tf.float32, shape=(None, n), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

predictions = layer(layer(X, n_inputs=n, n_outputs=n), n_inputs=n, n_outputs=1)
mse = tf.reduce_mean(tf.pow(predictions - y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_batch, y_batch = get_batch(scaled_data, housing.target.reshape(-1, 1), batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        if epoch % 100 == 0:
            error = sess.run(mse, feed_dict={X: scaled_data, y: housing.target.reshape(-1, 1)})
            print('Epoch {}, MSE = {}'.format(epoch, error))
