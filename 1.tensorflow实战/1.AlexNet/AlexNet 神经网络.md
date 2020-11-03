## AlexNet 神经网络

### 一 前言

AlexNet 深度卷积神经网络模型是Hinton的学生Alex提出来的，获得了2012年 ILSVRC 大赛的冠军。

训练数据集是 ImageNet，该数据集是华人美女学者 李菲菲 创办的，其中包含了大约1500万张图片，分为22000类，，每年的ILSVRC比赛大约有120万张图片，以及1000类的标注，可以说用到的数据集是ImageNet的子集，一般比赛中是以 top-5 和 top-1分类错误率作为模型性能的评测指标。

---

### 二 网络

AlexNet 神经网络一共有8层，前5层是全集层，后三层是全连接层。

模型输入的图片经过了数据增强，随机的从256 x 256的图片中截取 224 x 224大小的图片（同时还有水平镜面翻转）通过数据增强，最多可以增加 (256 - 224) ^ 2 x 2 = 2048 倍的数据量

关于每一层数据的 的输入输出 卷积核的大小、数量、移动步长、padding 等参数，这里就不再赘述，直接看代码就可以看的很清楚。

关于网络中的各部分的输入输出，以及最终程序运行出来的结果，可以运行代码查看输出 （显然不可能真的用比赛用的数据集来运行模型，此处仅仅是为了实现网络模型，模型输入的数据image也是随机创建的）

![image-20201103223159055](/home/dengruizhi/.config/Typora/typora-user-images/image-20201103223159055.png)



---

### 三 主要收获

1. Relu 激活函数，其效果在较深的神经网络中超过了Sigmod函数，解决了Sigmod函数梯度弥散的问题，Relu激活函数很早就提出，但是直到AlexNet 网络出现，才大放异彩。
2. LRN 局部归一化操作，现在几乎不再使用，其效果不明显，并且使  网络前馈、反馈的速度下降为原来的1 / 3，现在更多采用 Dropout，AlexNet网络中在后三层全连接网络中都采用了 Dropout。
3. AlexNet 之前的神经网络普遍采用 平均池化，AlexNet 中采用了最大池化，现在神经网络中较多使用 最大池化，并且AlexNet 中提出，池化核移动的步长应小于卷积核的大小，这样池化层的输出之间存在重叠和覆盖，提升了特征的丰富性。

4. **tf.name_scope() 和 tf.variable_scope() 函数 以及 tf.Variable() 和 tf.get_variable() 函数之间的区别**

~~~python
with tf.name_scope('conv1') as scope:
    var1 = tf.Variable(name='var', initial_value=1.0, dtype=tf.float32)
  
# 其中var1的name为 conv1/var:0，所以with tf.name_scope('conv1') as scope 的作用就是在其下面的所有变量name的前面添加前缀
# scope是一个string, 字符串的内容就是 conv1


import tensorflow as tf

var1 = tf.get_variable(name='conv1/var', dtype=tf.float32, initializer=1.0)
# var1 = tf.Variable(name='conv1/var', dtype=tf.float32, initial_value=1.0)

with tf.variable_scope('conv1', reuse=True) as scope:
    var2 = tf.get_variable(name='var', dtype=tf.float32, initializer=2.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('var1.name: ', var1.name)
    print('var2.name: ', var2.name)
    print(var1 is var2)
~~~

tf.name_scope() 和 tf.variable_scope() 函数的作用很类似，都是在变量的前面增加前缀，这样网络中不同层的变量的前缀不相同，同一层的变量的前缀相同，在`tensorboard` 中显示结点是会好看很多。

tf.name_scope() 和 tf.Variable() 搭配较多，tf.variable_scope() 和 tf.get_variable() 函数搭配较多，

后者的搭配更多用于实现 <font color = red>**变量共享**</font>，并且共享的变量除了要name相同，还要变量是tf.get_variable() 创建的。

重要的还有 `with tf.variable_scope('conv1', reuse=True) as scope: `中 reuse 的四种取值，False, True, None, tf.AUTO_REUSE 需要了解选用四种参数时各自对应的一些操作。

