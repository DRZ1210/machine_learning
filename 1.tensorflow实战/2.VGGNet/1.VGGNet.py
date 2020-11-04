#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from datetime import datetime
import time
import math



def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    '''
    Args:
        input_op: 输入的tensor, 如image
        name：这一层的名称
        kh: 卷积核的高度
        kw: 卷积核的宽度
        n_out: 卷积核的个数(输出通道数)
        dh: 步长的高
        dw: 步长的宽
        p: 参数列表, 包含每一层的卷积核和偏移量
    Function: 定义一个卷积层操作
    Returns: 卷积操作后的tensor
    '''
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope+'w', shape=[kh, kw, n_in, n_out], 
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        p += [conv, bias]
        return activation


def fc_op(input_op, name, n_out, p):
    '''
    Args:
        input_op: 输入的tensor  
        name: 该层的名称
        n_out: 输出通道数数
        p: 参数列表 包含每一层的权重矩阵和偏移量
    Function: 定义一个全连接操作
    Returns: 全连接操作后的tensor
    '''
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=name+'w', shape=[n_in, n_out], 
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # tf.nn.relu_layer(x, weight, bias) == tf.nn.relu(tf.matmul(x, weight) + bias)
        activation = tf.nn.relu_layer(input_op, kernel, bias, name=scope)
        p += [kernel, bias]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    '''
    Args: 
        input_op: 池化层的输入tensor
        name: 该层的名称
        kh: 池化核的高
        kw: 池化核的宽
        dh: 步长的高
        dw: 步长的宽
    Function: 定义最大池化层操作
    Returns: 最大池化操作后的tensor
    '''
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


# 开始搭建VGGNet-16网络

def inference_op(input_op, keep_prob):
    '''
    Args: 
        input_op: 输入的image的tensor 
        keep_prob: 是控制dropout比率的一个placeholder
    Function: 搭建整个VGGNet-16网络模型，前五部分为卷积部分，六部分为全连接部分(注意是部分，每一部分包含若干卷积层或全连接层)
    Returns: 
        predictions: 预测的输出 top1
        softmax: softmax操作后的输出
        fc8: 最后一层全连接网络的输出
        p: 参数列表
    '''
    p = []

    # 第一段卷积网络
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)

    # 第二段卷积网络
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)

    # 第三段卷积网络
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)

    # 第四段卷积网络
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)

    # 第五段卷积网络
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)

    # 第六段 全连接网络
    
    # 对pool5扁平化操作，reshape为一个vector
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')
    # 第一层全连接
    fc6 = fc_op(resh1, name='fc6', n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    # 第二层全连接
    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')
    # 第三层全连接
    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)

    return predictions, softmax, fc8, p



def time_tensorflow_run(sess, target, feed, info_string):
    '''
    Args:
        sess: tensorflow中的session
        target: 需要计算的算子
        feed: dropout中的keep_prob保留比率, 是一个placeholder
        info_string: 测试的信息
    Functions: 评估每轮计算的名称
    Returns: None
    '''
    # 预热轮数，前几轮迭代有显存加载、cache命中等问题，预热从而跳过这些问题
    num_step_burn_in = 10 
    # 总时间
    total_duration = 0.0
    # 平方和
    total_duration_squared = 0.0
    
    for i in range(num_step_burn_in + num_batches):
        start_time = time.time() 
        sess.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_step_burn_in:
            if not i % 10:
                # 每10次batch训练后，显示训练一次batch的时间
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_step_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration

    # mn: 每一轮迭代的平均耗时
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    # sd: 标准差
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch\n' % (datetime.now(), info_string, num_batches, mn, sd))



def run_benchmark():
    '''
    Args: None
    Function: 主函数，执行整个网络
    Returns: None
    '''
    with tf.Graph().as_default():
        image_size = 224
        # 采用随机生成的数据作为网络的输入，用ImageNet数据集训练根本不现实，至少在自己电脑上是这样 Wu~~~~~
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)

        print('done, congratulate!')
        # 计算机资源不行，下面一行不注销，显卡内存溢出报错
        # time_tensorflow_run(sess, grad, {keep_prob:0.5}, 'Forward-backward')


batch_size = 32
num_batches = 100

# 调用主函数，运行程序代码
run_benchmark()



