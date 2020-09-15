# coding=utf-8

import sys
import tensorflow as tf
import input_data
import numpy as np


mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# 关于图片的格式x: [-1, 5, 5, 1] 分别表示 图片数量无限、图片size为5x5、通道树为1
# 关于卷积核W: [5, 5, 1, 32] 分别表示 卷积核size为5x5、通道数为1(上一层输入的feature map的个数，若为图片则灰度图为1， RGB彩图为3)， 卷积核个数32个
# 关于池化窗口的大小 [batch, height, weight, channels] 首尾通常为1，中间的表示池化窗口的大小
# 关于Conv和pooling中的strides[1, 1, 1, 1] 首尾保持1不变，中间2，2表示卷积核(池化核)在x,y方向上移动的步长


def weight_variable(shape):
    """
    Args:
        shape: 张量变量(权重)的形状
    Function: 根据传入shape，返回截断的正态分布tensor变量
    Returns: tensor变量
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Args:
        shape: 张量变量(bias)的形状
    Function: 根据传入的变量，返回偏移量bias
    Returns: tensor 变量
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(layer, W):
    """
    Args:
        layer: 待处理信息(图片或中间结果)
        W: 卷积核
    Function: 对图片进行卷积操作
    Returns: 返回卷积后的feature maps
    """
    return tf.nn.conv2d(layer, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pooling_2x2(layer):
    """
    Args:
        layer: 上一层的feature map
    Function: 最大池化，池化核大小为2x2
    Returns: 返回池化后的结果
    """
    return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pooling_2x2(h_conv1)

# 第二层卷积池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])

W_fc1 = weight_variable([7 * 7 * 32, 256])
b_fc1 = bias_variable([256])

W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])

# 第一次全连接
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 第二次全连接
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

y = tf.nn.softmax(h_fc2)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 交叉熵损失函数
# 了解交叉熵损失函数的原理 (别忘了y的形状，若干行n，每行10列)
# tf.log(y) * y 对应位相乘，之后每一行求和(注意取负数)，得到n个数字的向量，之后求平均数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 梯度下降优化
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


training_iterations = 10000
batch_size = 50
display_step = 100
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./CNN", sess.graph)
    sess.run(tf.global_variables_initializer())

    for iteration in range(training_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, current_accuracy, _ = sess.run([merged, accuracy, optimizer], feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summary, iteration)

        if iteration % display_step == 0:
            print("Iteration: %d | Accuracy: %.6f" % (iteration, current_accuracy))

    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Test Accuracy: %.6f" % test_accuracy)
    writer.close()


