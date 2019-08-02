#-*- coding:utf-8 -*-
#neural network
#Author:tangyingjie
#选用不同优化算法进行测试



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error #均方误差MSE
from sklearn.metrics import mean_absolute_error #平方绝对误差MAE
from sklearn.metrics import r2_score            #R square

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
#y_data = np.array(lambde* mm_data[:,2]) + np.array(lambde* mm_data[:,3])

y_data = mm_data[:,2:5] # 表示最后两列的标签数据

#神经网络结构
batch_size = 400  #训练batch的大小
hide_size1 = 400 #隐藏神经元个数
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
cross_entropy =  tf.reduce_mean(tf.square(y_pred - y))                 #定义损失函数  均方误差 MSE
RMSE       =     tf.sqrt(tf.reduce_mean(tf.square(y_pred - y)))        #定义均方根误差RMSE
MAE        =     tf.reduce_mean(tf.abs(y_pred - y))                    #定义平均绝对误差MAE
#error       =     tf.abs((y))                                           #确定网络结构之后，用error作图

#R_Squared  = 1- tf.reduce_mean(tf.square(y_pred - y))/tf.reduce_mean(tf.square(y - tf.reduce_mean(y,0)))
#定义R Squared,R方，可决系数，是拟合优度的的一个统计量
R_Squared=1-tf.reduce_sum( tf.square(y - y_pred))/tf.reduce_sum( tf.square(y - tf.reduce_mean(y, 0)))

#train_step = tf.train.GradientDescentOptimizer (learning_rate=learning_rate).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer (learning_rate=learning_rate,momentum=0.5).minimize(cross_entropy)
#train_step = tf.train.AdadeltaOptimizer (learning_rate=learning_rate,rho=0.95).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.8,momentum=0.1).minimize(cross_entropy)#定义反向传播优化方法
#train_step = tf.train.AdagradOptimizer (learning_rate=learning_rate).minimize(cross_entropy)


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
        '''summary_str = sess.run(merged_summary, feed_dict=
        {x: x_data[int(len(matdata) * 0.85):len(matdata)],
        y: y_data[int(len(matdata) * 0.85):len(matdata)]})
        writer.add_summary(summary_str, i)'''
        #显示误差

        if i % 2000 == 0:
            print('训练%d次' % i )
            '''total_cross_entropy = (sess.run(cross_entropy,feed_dict =
            {x:x_data[int(len(matdata)*0.85):len(matdata)],
             y:y_data[int(len(matdata)*0.85):len(matdata)]}))
            #数据集后15%用作测试
            print('训练%d次后，误差为%f'%(i,total_cross_entropy))
            if total_cross_entropy <= 1e-4:
                break'''
    else:
        rmse,mae,r_squared = (sess.run([RMSE,MAE,R_Squared],feed_dict=
        {x: x_data,y:y_data}))
        print('RMSE为 %f  ,MAE为 %f  ,R_Squared为 %f  '%(rmse,mae,r_squared))

        #exit()
    n = np.random.randint(len(matdata))
    print(n)
    x_test = x_data[n]
    x_test = x_test.reshape(1, 2)
    predict_1 = sess.run(y_pred, feed_dict={x: x_test})

    #保存结果
    saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess,'./models',global_step=i)
    predict = sess.run(y_pred,feed_dict={x:x_data})
# predict = predict.ravel()#转换为向量
# orange = y_data.ravel()
orange = y_data
#建立时间轴
t = np.arange(len(matdata))
plt.plot(t,predict[:,0])
plt.plot(t,orange[:,0])
plt.show()

plt.plot(t,predict[:,1])
plt.plot(t,orange[:,1])
plt.show()

plt.plot(t,predict[:,2])
plt.plot(t,orange[:,2])
plt.show()

temp=np.zeros((1,4))
#temp[:,2:4] = predict_1
predict_1 = np.array([0,0,predict_1[0,0],predict_1[0,1],predict_1[0,2]]).reshape(1,5)
prediction = mm.inverse_transform(predict_1)
print("预测值" , prediction[:,2:5])

target = np.array([0,0,y_data[n][0],y_data[n][1],y_data[n][2]]).reshape(1,5)
target = mm.inverse_transform(target)
print("标签值：" , target[:,2:5])
