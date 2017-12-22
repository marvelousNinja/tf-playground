import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data')
n_inputs = 28 * 28
n_hidden_1 = 300
n_hidden_2 = 100
n_outputs = 10
n_epochs = 400
batch_size = 50

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

def dense_layer(X, name, n_inputs, n_outputs, activation=None):
    with tf.name_scope(name):
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_outputs), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_outputs]), name='biases')
        z = tf.matmul(X, W) + b
        return activation(z) if activation else z


with tf.name_scope('dnn'):
    layer_1 = dense_layer(X, 'layer_1', n_inputs, n_hidden_1, tf.nn.relu)
    layer_2 = dense_layer(layer_1, 'layer_2', n_hidden_1, n_hidden_2, tf.nn.relu)
    logits = dense_layer(layer_2, 'layer_3', n_hidden_2, n_outputs)

with tf.name_scope('loss'):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')

learning_rate = 0.01
training_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
        acc_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, 'Train acc:', acc_train, 'Test acc:', acc_test)

    save_path = saver.save(sess, './tf_logs/model.ckpt'))

## Making predictions
with tf.Session() as sess:
    saver.restore(sess, './tf_logs/model.ckpt')
    X_new_scaled = [] #...
    Z = sess.run(logits, feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
