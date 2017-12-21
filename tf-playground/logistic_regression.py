from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_moons

def add_variable_summary(variable, is_scalar=False):
    with tf.name_scope('summaries'):
        if is_scalar:
            tf.summary.scalar('{}_value'.format(variable.name), variable)
        else:
            mean = tf.reduce_mean(variable)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(variable))
            tf.summary.scalar('min', tf.reduce_min(variable))
            tf.summary.histogram('histogram', variable)

def scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def logistic_regression(X, n_features):
    with tf.name_scope('logistic_regression'):
        W = tf.Variable(tf.random_normal((n_features, 1)), name='weights')
        add_variable_summary(W)
        b = tf.Variable(0.0, name='bias')
        add_variable_summary(b, True)
        z = tf.add(tf.matmul(X, W), b, name='z')
        return 1.0 / (1.0 + tf.exp(-z))

def logloss(pred_y, true_y):
    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(
            true_y * tf.log(pred_y) + \
            (1 - true_y) * tf.log(1 - pred_y),
            name='logloss')
        add_variable_summary(loss, True)
        return loss

def get_batch(X, y, batch_size):
    indicies = np.random.choice(len(X), size=batch_size)
    return X[indicies], y[indicies]

data, target = make_moons(10000, noise=0.1, random_state=42)
scaled_data =  scale(data)
m, n = scaled_data.shape

n_epochs = 1000
batch_size = 100
num_batches = int(np.ceil(m / batch_size))
learning_rate = 0.01

X = tf.placeholder(tf.float32, (None, n), name='X')
y = tf.placeholder(tf.float32, (None, 1), name='y')
pred = logistic_regression(X, n)
loss = logloss(pred, y)
training_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

merged_summary = tf.summary.merge_all()
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = "{}/run-{}".format(root_logdir, now)
train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
model_path = root_logdir + '/model.ckpt'
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Or restore the model
    # saver.restore(sess, model_path)
    for epoch in range(n_epochs):
        for batch in range(num_batches):
            X_batch, y_batch = get_batch(scaled_data, target.reshape(-1, 1), batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 50 == 0:
            summary, error = sess.run([merged_summary, loss], feed_dict={X: scaled_data, y: target.reshape(-1, 1)})
            print('Epoch {}, Logloss = {}'.format(epoch, error))
            train_writer.add_summary(summary, epoch)
            saver.save(sess, model_path)
