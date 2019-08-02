#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
s_init = 6.425

data = np.loadtxt("./test/data/优化数据7.31/6.425.txt")   #将文件中数据加载到data数组里
plt.xlabel(' Response Time(s=%f)' % s_init, fontsize=15)
plt.ylabel('Hysteresis Band(s=%f)' % s_init, fontsize=15)
plt.scatter(data[:,2], data[:,3])
plt.savefig('./plot/8.1/'+str(round(s_init,3))+'.png')
plt.show()
print("safaslfks;fas")