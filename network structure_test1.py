#-*- coding:utf-8 -*-
#neural network_test
#Author:tangyingjie
#此程序用于训练测试最佳网络结构（1个隐含层）
#结果保存在./test/data/1.txt文本中



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import random
import math
import xlsxwriter
from sklearn.preprocessing import MinMaxScaler
import csv
import codecs


global mm,mm_data,matdata
mm= MinMaxScaler()
path = r'./dataset/dataset.mat'
matdata = scio.loadmat(path)['final_result']
mm_data = mm.fit_transform(matdata)              #归一化

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'   '   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.write('\n')
    file.close()
    print("保存文件成功")

for times in range(400,401):
    #生成输入输出数据
    input_size =  2 #输入变量个数
    output_size = 3#输出变量个数
    data_size = len(matdata)#样本个数
    lambde = 0.5
    x_data = mm_data[:, :input_size]  # X_data为前2列特征数据

    y_data = np.zeros((data_size,output_size))#初始化输出数据
    #y_data = np.array(lambde* mm_data[:,2]) + np.array(lambde* mm_data[:,3])

    y_data = mm_data[:,2:5] # 表示最后两列的标签数据

    #神经网络结构
    batch_size = 400  #训练batch的大小
    hide_size1 = times #隐藏神经元个数
    #hide_size2 = 70 #隐藏神经元个数
    #预留每个batch中输入与输出的空间
    x = tf.placeholder(tf.float32,shape = (None,input_size),name='x')     #None随batch大小变化
    y = tf.placeholder(tf.float32,shape = (None,output_size),name='y')

    w_hidden1 = tf.Variable(tf.random_normal([input_size,hide_size1],stddev = 1,seed = 1,name='w1'))
    b_hidden1 = tf.Variable(tf.zeros([1,hide_size1],dtype = tf.float32),name='b1')
    h1 = tf.nn.tanh(tf.matmul(x,w_hidden1)+b_hidden1)

    w_output = tf.Variable(tf.random_normal([hide_size1,output_size],stddev = 1,seed = 1))
    y_pred = tf.nn.sigmoid(tf.matmul(h1,w_output),name='y_pred')

    #反向损失函数
    learning_rate = 0.001#学习速率
    cross_entropy =  tf.reduce_mean(tf.square(y_pred - y))        #定义损失函数  均方误差 MSE
    RMSE       =     tf.sqrt(tf.reduce_mean(tf.square(y_pred - y)))        #定义均方根误差RMSE
    MAE        =     tf.reduce_mean(tf.abs(y_pred - y))         #定义平均绝对误差MAE





    #R_Squared  = 1- tf.reduce_mean(tf.square(y_pred - y))/tf.reduce_mean(tf.square(y - tf.reduce_mean(y,0)))
    #定义R Squared,R方，可决系数，是拟合优度的的一个统计量
    R_Squared=1-tf.reduce_sum( tf.square(y - y_pred))/tf.reduce_sum( tf.square(y - tf.reduce_mean(y, 0)))

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)



    #创建会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)#初始化变量
        tf.summary.scalar("MSE", cross_entropy)
        tf.summary.scalar("RMSE", RMSE)
        tf.summary.scalar("MAE", MAE)
        tf.summary.scalar("R_Squared", R_Squared)
        #tf.summary.scalar("AARE", AARE)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/", sess.graph) # tensorboard可视化操作
        #设定训练次数
        STEPS = 100000#训练次数
        for i in range(STEPS):
            #选取训练batch
            start = max((i * batch_size) % int(len(matdata)*0.7),0)
            end = min(start + batch_size,int(len(matdata)*0.7))#取前85%进行训练
            #计算
            _,a=sess.run([train_step,R_Squared],feed_dict = {x:x_data[start:end],y:y_data[start:end]})
            #sdas=sess.run(R_Squared,feed_dict={x:x_data[start:end],y:y_data[start:end]})
            #显示误差

            if i % 2000 == 0:
                print('训练%d次' % i )

        rmse,mae,r_squared = (sess.run([RMSE,MAE,R_Squared],feed_dict=
        {x: x_data,y:y_data}))
        data = [str(input_size)+'×'+str(hide_size1)+'×'+str(output_size),mae,rmse,r_squared]
        print('RMSE为 %f  ,MAE为 %f  ,R_Squared为 %f  '%(rmse,mae,r_squared))
        text_save('./test/data/1.txt',data)

print('操作完成')