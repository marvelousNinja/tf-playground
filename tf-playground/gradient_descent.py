import os
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

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

theta = tf.Variable(tf.random_uniform((n + 1, 1), -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
# Manual gradient
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# Autodiff
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# TF GD Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# TF Momentum
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

def fetch_batch(X, y, batch_index, batch_size):
    indices = np.random.choice(X.shape[0], size=batch_size)
    return X[indices], y[indices]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(housing_with_bias, housing.target.reshape(-1, 1), batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE=', mse.eval(feed_dict={X: X_batch, y: y_batch}))
            print(theta.eval())

    saver.save(sess, '/tmp/session_final.ckpt')
    best_theta = theta.eval()
    print(best_theta)
