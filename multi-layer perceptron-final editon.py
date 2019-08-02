#-*- coding:utf-8 -*-
#multi-layer perceptron-final editon
#Author:tangyingjie
#用于生成网络模型 以及tensorboard数据
#并用于通过损失函数确定迭代次数
#采用的最终版本

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import random
import math
from sklearn.preprocessing import MinMaxScaler


global mm,mm_data,matdata
mm= MinMaxScaler()
path = r'./dataset/dataset.mat'
matdata = scio.loadmat(path)['final_result']
mm_data = mm.fit_transform(matdata)              #归一化


#生成输入输出数据
input_size =  2 #输入变量个数
output_size = 3#输出变量个数
data_size = len(matdata)#样本个数
lambde = 0.5
x_data = mm_data[:, :input_size]  # X_data为前2列特征数据
y_data = np.zeros((data_size,output_size))#初始化输出数据
keep_prob = tf.placeholder(tf.float32)

#y_data = np.array(lambde* mm_data[:,2]) + np.array(lambde* mm_data[:,3])

y_data = mm_data[:,2:5] # 表示最后两列的标签数据


def add_layer(input,in_size,out_size,n,activate=None):
    layer_name = 'layer'+str(n)
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size],stddev = 1,seed = 1,name='W'))
            tf.summary.histogram('weight',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size], dtype=tf.float32), name='B')
            tf.summary.histogram('biases',biases)
        with tf.name_scope('matmul'):
            Wx_plus_b = tf.matmul(input, Weights) + biases
        with tf.name_scope('outputs'):
            outputs = activate(Wx_plus_b,name='y_pred')
            tf.summary.histogram('outputs',outputs)
        return outputs


batch_size = 200  #训练batch的大小
#神经网络结构
#预留每个batch中输入与输出的空间
x = tf.placeholder(tf.float32,shape = (None,input_size),name='x')     #None随batch大小变化
y = tf.placeholder(tf.float32,shape = (None,output_size),name='y')

L1 = add_layer(x,2,22,1,tf.nn.tanh)
L2 = add_layer(L1,22,196,2,tf.nn.tanh)
y_pred = add_layer(L2,196,3,3,tf.nn.sigmoid)


#反向损失函数
learning_rate = 0.001#学习速率
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.square(y_pred - y))
    Loss = tf.summary.scalar("loss", cross_entropy)
with tf.name_scope('MSE'):
    mse = tf.reduce_mean(tf.square(y_pred - y))  # 定义损失函数  均方误差 MSE
    Mse = tf.summary.scalar("MSE", mse)
    mse_1 = tf.summary.scalar("MSE_1", cross_entropy)
    mse_2 = tf.summary.scalar("MSE_2", cross_entropy)
with tf.name_scope('RMSE'):
    RMSE       =     tf.sqrt(tf.reduce_mean(tf.square(y_pred - y)))        #定义均方根误差RMSE
    Rmse = tf.summary.scalar("RMSE", RMSE)
with tf.name_scope('MAE'):
    MAE        =     tf.reduce_mean(tf.abs(y_pred - y))         #定义平均绝对误差MAE
    Mae = tf.summary.scalar("MAE", MAE)
with tf.name_scope('R_Squared'):
    R_Squared=1-tf.reduce_sum( tf.square(y - y_pred))/tf.reduce_sum( tf.square(y - tf.reduce_mean(y, 0)))
    R_squared = tf.summary.scalar("R_Squared", R_Squared)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
# tf.summary.scalar("AARE", AARE)

#创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)#初始化变量
    merged_summary = tf.summary.merge_all()
    # merged_summary = tf.summary.merge([Mse,Rmse,Mae,R_squared])
    # merge_summary_1 = tf.summary.merge([mse_1])  # 这里的[]不可省
    # merge_summary_2 = tf.summary.merge([mse_2])  # 这里的[]不可省
    writer = tf.summary.FileWriter("log/", sess.graph) # tensorboard可视化操作
    #设定训练次数
    STEPS = 87000#训练次数
    for i in range(STEPS):
        #选取训练batch
        start = max((i * batch_size) % int(len(matdata)*0.7),0)
        end = min(start + batch_size,int(len(matdata)*0.7))#取前70%进行训练
        _,a=sess.run([train_step,R_Squared],feed_dict = {x:x_data[start:end],y:y_data[start:end]})
        if i %50 ==0:
            result = sess.run(merged_summary,feed_dict={x:x_data[start:end],y:y_data[start:end]})
            writer.add_summary(result, i)
            print('训练%d次' % i)
        #显示误差i
    rmse,mae,r_squared = (sess.run([RMSE,MAE,R_Squared],feed_dict={x: x_data,y: y_data}))
    print('测试集：RMSE为 %f  ,MAE为 %f  ,R_Squared为 %f  ' % (rmse, mae, r_squared))
    saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess,'./models/models.ckpt',global_step=i)
    predict = sess.run(y_pred,feed_dict={x:x_data})


