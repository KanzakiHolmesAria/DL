#! /usr/bin/python
# -*- coding: UTF-8 -*-
# Author: UMR
import tensorflow as tf
import input_data
'''
导入MNIST数据集
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
Softmax回归模型
'''
X = tf.placeholder('float', [None, 784])  # 输入用占位符placeholder, None表示可以为任意值
W = tf.Variable(tf.zeros([784, 10]))  # 权重用变量Variable
b = tf.Variable(tf.zeros([10]))  # 偏置用变量Variable
y = tf.nn.softmax(tf.matmul(X, W) + b)  # 调用nn方法的softmax(z)函数， z = WX + b
'''
cross-entropy交叉熵损失函数
'''
y_ = tf.placeholder('float', [None, 10])  # 输出（真实标签）用占位符placeholder, None表示可以为任意值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
'''
反向传播算法
'''
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)  # 定义训练使用学习率0.01的梯度下降最小化交叉熵
init = tf.global_variables_initializer()  # 所有变量Variables在使用前必须进行初始化
sess = tf.Session()  # tf中运算必须通过session执行
sess.run(init)  # 初始化变量
for i in range(1000):  # 模型循环训练次数：1000
    batch_xs, batch_ys = mnist.train.next_batch(100)  # next_batch是mnist数据集中定义好的随机选取100个样本的函数，实际使用中需要自行定义
    sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})  # feed_dict用于填补占位符
'''
模型评估:使用accuracy作为指标
'''
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))  # argmax(y_, 1)用于获取y_中每行的最大值的下标索引，argmax(y_, 0)用于列
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))
sess.close()
