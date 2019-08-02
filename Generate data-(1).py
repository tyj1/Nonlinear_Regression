#-*- coding:utf-8 -*-
#Generate data-(1)
#Author:tangyingjie
#加载 multi-layer perceptron模型，生成实验测试数据predicted_value_new.txt
# 以及预测数据experimental_data_new.txt


#Importing required modules
import math
import random
import scipy.io as scio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

mm= MinMaxScaler()
path = r'./dataset/dataset10.mat'

matdata = scio.loadmat(path)['final_result']
matdata_test = matdata[int(len(matdata)*0.55):int(len(matdata)*0.6)]   #测试用数据


mm_data = mm.fit_transform(matdata)              #归一化
result_1 = mm.inverse_transform(mm_data)

x_data = mm_data[:, :2]  # X_data为前2列特征数据


x_data_test = x_data[int(len(matdata)*0.55):int(len(matdata)*0.6)]     #测试用数据


#加载模型
sess = tf.Session()
X = None # input
yhat = None # output
modelpath = r'./models/'
saver = tf.train.import_meta_graph(modelpath + 'models.ckpt-86999.meta')   #此处加载模型，请一定选取正确的模型
saver.restore(sess, tf.train.latest_checkpoint(modelpath))
graph = tf.get_default_graph()
# for n in tf.get_default_graph()._nodes_by_name:    #  用来打印graph中的nodes，查看其中tensorflow名
#     print (n)
X = graph.get_tensor_by_name("x:0")
yhat = graph.get_tensor_by_name("layer3/outputs/y_pred:0")
print('Successfully load the pre-trained model!')


temp = sess.run(yhat, feed_dict={X: x_data_test})  # (-1, 3)
result_final = np.hstack([x_data_test,temp])
temp1 = sess.run(yhat, feed_dict={X: x_data})

result_fina3 =  np.hstack([x_data,temp1])


mm= MinMaxScaler()
mm_data = mm.fit_transform(matdata)              #归一化
result_final2 = mm.inverse_transform(result_final)
result_final4 = mm.inverse_transform(result_fina3)
np.savetxt("./test/data/predicted_value_new.txt", result_final2,fmt='%f',delimiter=',')

np.savetxt("./test/data/experimental_data_new.txt", matdata_test,fmt='%f',delimiter=',')

#np.savetxt("./test/data/4.txt", matdata_test,fmt='%f',delimiter=',')   #测试用
print('sfhujashgfokjasg法师法师法师发大水f')




