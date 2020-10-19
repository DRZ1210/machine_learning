#coding=utf-8


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 使用scikit-learn 中的preprocessing 模块
import sklearn.preprocessing as prep



def xavier_init(fan_in, fan_out, constant = 1):
    """
    Args: fan_in: 输入结点的个数、fan_out: 输出结点的个数、constant 常数
    Function: xavier 初始化
    Return: 返回满足Ex = 0, Dx = 2 / (fan_in + fan_out) 的分布，此处采用均匀分布
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval = low, maxval = high, 
                             dtype = tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    # 去噪自编码器 采用的噪声是加性高斯噪声
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """
        Args: 
            n_input: 输入变量的数量
            n_hidden: 隐藏层结点数量
            transfer_function: 隐藏层激活函数，默认为softplus （softplus 认为是Relu函数的平滑化版本）
            optimizer: 优化器 默认是 Adam，是梯度下降算法的变种，Adam源自自适应矩估计 adaptive monent estimation
            scale: 高斯噪声系数 默认为0.1 
            network_weights: 参数初始化，采用后面定义的函数
        Function: 构造函数
        Returns: None
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)  # 测试的额高斯噪声系数 
        self.training_scale = scale  # 训练的高斯噪声系数 默认为0.1
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        # 开始定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 定义隐藏层 ((x + 噪声) * 权重 + 偏移量) + 激活函数
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        # 对隐藏层输出层 进行数据复原、重建操作
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 自编码器的损失函数，采用平方和
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x, self.reconstruction), 2.0))
    
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        Args: None
        Function: 参数初始化函数
        Returns: 返回字典，包含hidden层和reconstruction需要的权重和偏移量
        """
        all_weights = dict();
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_input), dtype = tf.float32)
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights


    def partial_fit(self, x):
        """
        Args: 
            x: 网络最早的输入
        Function: 使用1 batch的数据训练并返回当前的损失cost
        Returns: 损失cost
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        """
        Args:
            x: 网络的输入
        Function: 函数在测试集中用于模型性能评测，用于计算cost，注意是测试集中使用
        Returns: 损失cost
        """
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})
        
    def transform(self, X):
        """
        Args: 
            x: 网络的输入
        Function: 获取抽象后的特征，自编码器的隐藏层的最主要的功能就是学习数据中的高阶特征
        Returns: 返回自编码器隐藏层的输出结果
        """
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden = None):
        """
        Args: 
            hidden: 隐藏层的输出
        Function: 重建层将提取到的高阶特征复原为原始数据
        Returns: self.reconstruction 原始数据
        """
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """
        Args:
            x: 网络的输入
        Function: 相当于transform()和generate()函数两者综合
        Returns: 输入是原数据，返回的是复原后的数据
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        """
        Args: None
        Function: 获取隐藏层的权重w1
        Returns: 返回权重w1
        """
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """
        Args: None
        Function: 获取隐藏层的偏移量b1
        Return: 返回偏移量b1
        """
        return self.sess.run(self.weights['b1'])


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


# 对数据进行标准化处理，就是将数据转换为 均值为0，标准差为1的分布
# 标准化的方法就是 先减去均值，再除以标准差


def standard_scale(X_train, X_test):
    """
    Args:
        X_train: 训练数据
        X_test: 测试数据
    Function: 将训练数据和测试数据进行标准化处理
    Returns: 返回标准化之后的 训练数据和测试数据
    """
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    """
    Args: 
        data: 测试或训练的数据
        batch_size: 每一次拿的数据的多少, batch
    Function: 从data中随机选取batch_size大小的数据，采用不放回抽样
              方法是从[0, len(data) - batch_size]中随机选取一个整数，并一次为起点向后选取batch_size大小的数据
    Returns: data中batch_size大小的数据
    """
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


# 调用函数对数据进行标准化处理
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 宗训练样本数
n_samples = int(mnist.train.num_examples)

# 最大训练的轮数
training_epoch = 20 

# 每次拿取的数据的大小
batch_size = 128

# 展示cost间隔的轮数
display_step = 1


# 实例化AGN 自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, 
                                              n_hidden = 200,
                                              transfer_function = tf.nn.softplus,
                                              optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                              scale = 0.01)



for epoch in range(training_epoch):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))



print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))



'''
程序输出如下：
Epoch:  0001 cost =  25091.606385227
Epoch:  0002 cost =  14429.040821591
Epoch:  0003 cost =  11738.597227273
Epoch:  0004 cost =  10490.628647727
Epoch:  0005 cost =  9403.630959091
Epoch:  0006 cost =  8822.638245455
Epoch:  0007 cost =  9551.524561364
Epoch:  0008 cost =  8784.300501136
Epoch:  0009 cost =  8873.799493750
Epoch:  0010 cost =  8970.831257386
Epoch:  0011 cost =  8234.742071591
Epoch:  0012 cost =  8096.298430682
Epoch:  0013 cost =  8414.537575000
Epoch:  0014 cost =  7955.590493750
Epoch:  0015 cost =  7705.224820455
Epoch:  0016 cost =  7949.962841477
Epoch:  0017 cost =  8445.097561364
Epoch:  0018 cost =  7952.284888636
Epoch:  0019 cost =  7917.255268750
Epoch:  0020 cost =  7037.569138636
Total cost: 689717.75

real	1m4.294s
user	4m29.948s
sys	0m7.254s
'''


