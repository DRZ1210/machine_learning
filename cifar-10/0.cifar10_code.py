#!/usr/bin/env python
# coding=utf-8


import tensorflow as tf
import numpy as np
import time
import math
import cifar10, cifar10_input
import matplotlib.pyplot as plt

batch_size = 128
max_steps = 4000
data_dir = './1.cifar10_data/cifar-10-batches-bin/'


'''
在分类和和回归问题中，特征过多会导致过拟合，为了减少过拟合可以 减少特征的数量，或者惩罚不重要的特征的权重
在说L1、L2正则之前，首先需要知道 范数的概念以及范数的计算公式
L1正则化，会制造稀疏的特征，即将不重要的特征的权重置为0
L2正则化，使特征的权重不会过大，比较平均
'''
def variable_with_loss(shape, stddev, w1):
    '''
    Args:
        shape: 卷积核的形状
        stddev: 正态分布的标准差
        w1: 用于控制权重 L2 loss的大小
    
    Function: 用tf.truncated_normal()截断的正态分布初始化权重，并做一个L2正则化处理
              将权重的损失加入到 losses 列表中
    Returns: 权重 weight
    '''
    var =  tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name = 'weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# 使用cifar10类下载数据集，解压，展开到默认位置
cifar10.maybe_download_and_extract()


# 生成训练使用的 images and labels 
# data_dir: 表示cifar10 的数据路径
# batch_size: 表示每一次从数据集中拿出数据的数量
# 对数据进行了数据增强 Data Augmentation，增加样本量，减少过拟合, 并对数据进行标准化
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)


# 生成测试使用的images andd labels
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

print("training image:")
print(type(images_train))
print(images_train.shape)
print(images_train[0].shape)

print("tranging labels:")
print(type(labels_train))
print(labels_train.shape)


# 开始搭建模型
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层 卷积层
weight1 = variable_with_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=0.1, alpha=0.001 / 9.0, beta=0.75)

# 第二层 卷积层
weight2 = variable_with_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第三层 全连接层, 全连接输出的结点个数为384
# 将pool2 flatten 扁平化，每一张图片变为一维向量
# 全连接层，为防止过拟合，对权重参数进行惩罚，设置了一个weight loss, 使weight中的所有参数都收到L2约束
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第四层 全连接层，同上，只是输出结点的数量减半为192
weight4 = variable_with_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层 全连接层，输出结点为10，也就是最后的分类
# 最后一层的权重stddev大小为上一层输出结点的倒数，并且权重没有惩罚
weight5 = variable_with_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.1, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# 定义模型的损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    # 传入logits, labels 计算两者之间的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # 将 cross_entropy 的损失加入到整体的 loss 中（还有对于weight 的惩罚损失）
    tf.add_to_collection('losses', cross_entropy_mean)
    # 将损失列表中的所有项加和，并返回
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# 调用损失函数
loss = loss(logits, label_holder)

# 定义优化器
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# logits 是一个二维的矩阵，每一行表示一个image，一行中对于10种中的每一种都有一个概率
# 参数k == 1， 取概率最大的值对应的下标 和 label中的值进行比较，若相同，返回Ture, 否则返回False
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# 创建sess, 并初始化所有的变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 之前对进行数据增强操作，这里采用了16个线程加速操作
tf.train.start_queue_runners()

# 开始训练数据 max_steps 轮
# 每10 step 展示训练的损失、一个batch 训练的时间，每一秒训练的examples个数
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([optimizer, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        example_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        print('step %d, loss=%.2f (%.1f example/sec; %.3f sec/batch)' % 
                    (step, loss_value, example_per_sec, sec_per_batch))


# 在测试集上进行测试
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_examples:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    if step % 50 == 0:
        print('step: %d has tested' % step)
    step += 1


precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)








