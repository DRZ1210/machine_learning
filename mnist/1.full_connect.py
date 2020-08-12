# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)


def show_picture(img):
    plt.imshow(np.reshape(img, [28, 28]))
    plt.show()


def show_mnist_info():
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)
    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)
    for i in range(5):
        show_picture(mnist.train.images[i])


# show_mnist_info()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# create model
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w_fc1 = weight_variable([784, 200])
b_fc1 = bias_variable([200])
w_fc2 = weight_variable((200, 200))
b_fc2 = bias_variable([200])
w_out = weight_variable([200, 10])
b_out = bias_variable([10])

hidden1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, w_fc2) + b_fc2)
y_pred = tf.nn.softmax(tf.matmul(hidden2, w_out) + b_out)


# loss and optimizer
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)), tf.float32))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=1))
learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# session
display_step = 100
batch_size = 100
training_iterations = 10000

tf.summary.scalar("Accuracy", accuracy)
tf.summary.scalar("Cross entropy", cross_entropy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./network", sess.graph)
    sess.run(tf.global_variables_initializer())

    for iteration in range(training_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, current_accuracy, _ = sess.run([merged, accuracy, optimizer], feed_dict={x: batch_xs, y: batch_ys})
        writer.add_summary(summary, iteration)

        if iteration % display_step == 0:
            print("Iteration: %d | Accuracy: %.6f" % (iteration, current_accuracy))

    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Test Accuracy: %.6f" % test_accuracy)
    writer.close()


