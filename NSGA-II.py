# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: tangyingjie
'''说明 ：
    采用训练好的模型，进行多目标优化，结果保存在./test/data/优化数据1中
    图片保存在./plot/7.28中

'''

#Importing required modules
import math
import random
import scipy.io as scio
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

path = r'./dataset/dataset.mat'

matdata = scio.loadmat(path)['final_result']


#Function to find index of list   #寻找列表的索引
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values   #根据值排序
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#快速排序算法
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    global population,s_left,s_right,s_init

    if mutation_prob <1:
        flagx = 0
        #此处为变异操作，将变异限制为固定
        Temp2 = []
        while(flagx!=1):
            solution_1 = np.array([random.random()]).reshape(1, -1)  # 随机产生种群，生成初始种群
            solution_2 = np.array([random.random()]).reshape(1, -1)  # 随机产生种群，生成初始种群
            Temp = np.hstack((solution_1, solution_2))
            flag = sess.run(yhat, feed_dict={X: Temp})  # (-1, 3)
            # 反归一化
            Temp1 = np.hstack([solution_1, solution_2, flag])
            Flag = Inverse_normalization_function(matdata,Temp1)
            x = np.equal(population, Temp).astype(int).sum(axis=1).reshape(1,-1)
            xxasdas = np.where(x == 2)
            sdasfas = any(xxasdas)
            if (((s_init-lens) < Flag[:, 4] < (s_init+lens)) and (bool(1-sdasfas))):
                Temp2.append(list(Temp))
                asdas = np.array(Temp2).reshape(1, 2)
                flagx = 1
    return asdas
def geberate_initpop():
    global pop_setnum
    Temp2 = []
    while (pop_setnum < pop_size):
        solution_1 = np.array([random.random()]).reshape(1, -1)  # 随机产生种群，生成初始种群
        solution_2 = np.array([random.random()]).reshape(1, -1)  # 随机产生种群，生成初始种群
        Temp = np.hstack((solution_1, solution_2))
        flag = sess.run(yhat, feed_dict={X: Temp})  # (-1, 3)
        Temp1 = np.hstack([solution_1, solution_2, flag])
        Flag = Inverse_normalization_function(matdata,Temp1)

        if (s_left < Flag[:, 4] < s_right):
            pop_setnum = pop_setnum + 1
            Temp2.append(list(Temp))
        if (pop_setnum == pop_size):
            result = np.array(Temp2).reshape(pop_size, -1)


    pop_setnum = 0
    return result
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'   '   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.write('\n')
    file.close()
    print("保存文件成功")
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

#加载模型
sess = tf.Session()
X = None # input
yhat = None # output
modelpath = r'./models/'
saver = tf.train.import_meta_graph(modelpath + 'models.ckpt-86999.meta')   #此处加载模型，请一定选取正确的模型
saver.restore(sess, tf.train.latest_checkpoint(modelpath))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("x:0")

yhat = graph.get_tensor_by_name("layer3/outputs/y_pred:0")

print('Successfully load the pre-trained model!')



#Main program starts here
pop_size = 25
max_gen = 200

#Initialization
min_x1=50            #要进行修改
max_x1=1000             #要进行修改
min_x2=0
max_x2=80
lens = 0.025
s_init = 6.425            #阀芯位移条件限制
s_limit = 8
s_left = s_init - lens
s_right = s_init + lens
Temp2 = []
gen_no = 0
pop_setnum = 0
while(s_init < (6.425+0.001)):
#while(s_init < s_limit):
    population = geberate_initpop()    #生成初始种群

    while(gen_no<max_gen):

        function_values = sess.run(yhat, feed_dict={X: population}) # (-1, 3)
        function1_values = list(function_values[:,0])
        function2_values = list(function_values[:, 1])
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
        print("The best front for Generation number ",gen_no, " is")
        for valuez in non_dominated_sorted_solution[0]:
            print(np.round(population[valuez,:],3),end=" ")
        print("\n")
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = population

        #Generating offsprings
        while(len(solution2)!=2*pop_size):
            a1 = random.randint(0,pop_size-1)

            c = np.array(mutation(population[a1, :])).reshape(1, 2)
            #c = np.array(crossover(population[a1,:],population[b1,:])).reshape(1,2)    #
            solution2 =  np.append(solution2,c, axis=0)
        function_values2 = sess.run(yhat, feed_dict={X: solution2}) # (-1, 3) #对40个变量进行求解，获得40个解集
        function1_values2 = list(function_values2[:,0])
        function2_values2 = list(function_values2[:, 1])
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:]) #对其进行排序
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()

            for value in front:
                new_solution.append(value)
                if(len(new_solution)==pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        population = [solution2[i] for i in new_solution]
        population = np.array(population)
        gen_no = gen_no + 1
    gen_no = 0

    #Lets plot the final front now
    function1 = [i  for i in function1_values]
    function2 = [j  for j in function2_values]

    temp = sess.run(yhat, feed_dict={X: population})  # (-1, 3)
    temp.astype(np.float64)
    d=np.hstack([population,temp])


    result = Inverse_normalization_function(matdata,d)
    plt.xlabel(' Response Time(s=%f)'%s_init, fontsize=15)
    plt.ylabel('Hysteresis Band(s=%f)'%s_init, fontsize=15)
    plt.scatter(result[:,2], result[:,3])

    plt.savefig('./plot/7.31/'+str(round(s_init,3))+'.png')
    plt.show()
    np.savetxt("./test/data/优化数据7.31/" + str(round(s_init, 3)) + ".txt", result,fmt='%f',delimiter=',')
    #text_save('./test/data/优化结果(第8个点)_test.txt', result[7,:])
    s_init =s_init + lens
print("操作完成")

