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
path = r'./dataset/dataset.mat'

matdata = scio.loadmat(path)['final_result']


#归一化函数
def Normalized_function(narray1):     #narray为二位数组
    max=[]
    min=[]
    for i in range(narray1.shape[1]):
        max.append(np.amax(narray1[:, i]))
        min.append(np.amin(narray1[:, i]))
        if(i==0):
            Normalized_narray = np.array(list((narray1[:,i]-min[i])/(max[i] - min[i]))).reshape(-1,1)
        else:
            Normalized_narray = np.append(Normalized_narray,
                                   np.array(list((narray1[:,i]-min[i])/(max[i] - min[i]))).reshape(-1,1),
                                   axis=1)
    #max = np.array(max).reshape(1,-1)
    #min = np.array(min).reshape(1, -1)
    return Normalized_narray,max,min

#反归一化函数
def Inverse_normalization_function(narray1,narray2):
    _,max,min = Normalized_function(narray1)
    for i in range(narray1.shape[1]):
        if(i==0):
            narray3 = np.array(narray2[:,i]*(max[i]-min[i]) + min[i]).reshape(-1,1)
        else:
            narray3 = np.append(narray3,
                                np.array(narray2[:,i]*(max[i]-min[i]) + min[i]).reshape(-1,1),
                                axis=1)
    return narray3
solution_1=np.random.rand(1, 5)     #生成0到10区间，3行4列的随机数组
#Normalized_narray,test1,test2 = Normalized_function(matdata)
#xxxxxxx = Inverse_normalization_function(matdata,solution_1)

a=np.array(range(9)).reshape(3,3)
b=np.array(range(3)).reshape(1,3)

x = np.equal(a,b).astype(int).sum(axis=1).reshape(1,-1)




print(x.sum(axis=1))



