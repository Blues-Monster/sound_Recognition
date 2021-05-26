import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math



#----------------参数-------------------
train_times = 10000          #训练次数
hiddennodes0 = 306
hiddennodes1 = 153
data_path = './music_data/data.csv'
learningrate = 0.01
#---------------------------------------





#--------特征规范化----------
#提取数据文件
music_dict = {'blues':0,'classical':1,'country':2,'disco':3,'hiphop':4,'jazz':5,'metal':6,'pop':7,'reggae':8,'rock':9}
data = pd.read_csv(data_path,encoding= 'utf-8')
label = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
y =[music_dict[i] if i in music_dict else i for i in label]
#对矩阵进行归一化
trans_X = [0.00001,100000,100,100000,1000,0.01,0.001,0.00001,0.001,0.000001,0.01,0.01]+[0.1]*3+[1]+[0.1]+[1]+[0.1]+[1]*2+[1]*9+[0.001]+[0.01]*20+[100]*23+[10]+[0.01]
X = np.dot(X,np.diag(trans_X))
#拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=314)


#------------生成神经网络中间层--------------
input_nodes = X_train.shape[1]
train_number = X_train.shape[0]
output_nodes = len(music_dict)

def random_range(x,y):return math.sqrt(6/(x+y))

w0_range = random_range(input_nodes,hiddennodes0)
w0 = np.random.rand(input_nodes,hiddennodes0)*(2*w0_range)-w0_range
w1_range = random_range(hiddennodes0,hiddennodes1)
w1 = np.random.rand(hiddennodes0,hiddennodes1)*(2*w1_range)-w1_range
w2_range = random_range(hiddennodes1,output_nodes)
w2 = np.random.rand(hiddennodes1,output_nodes)*(2*w2_range)-w2_range

#偏导
w0_ = w0
w1_ = w1
w2_ = w2

#-----------标准输出矩阵------------
oneHot = np.identity(len(music_dict))
y_true = oneHot[y_train]

#------------激活函数---------------
'''
'identity':无激活操作，有助于实现线性瓶颈, 返回 f(x) = x
'logistic':逻辑函数, 返回 f(x) = 1 / (1 + exp(-x))
'tanh': 双曲线函数, 返回 f(x) = tanh(x)
'relu': 矫正线性函数, 返回 f(x) = max(0, x)，（默认）
'''
def activation_function(x):
    #if(x>0):
    return 1/(1+np.exp(-0.1*x))
    #else:
    #    return 1/(1+np.exp(0.01*x))
    #return x


for times in range(train_times):
    #-----------正向传递---------------
    hide0 = np.dot(X_train,w0)
    act0 = activation_function(hide0)

    hide1 = np.dot(act0,w1)
    act1 = activation_function(hide1)

    out1 = np.dot(act1,w2)
    prediction = activation_function(out1)




    
    #----------反向传播------------------

    loss_out = y_true-prediction
    loss_1 = np.dot(loss_out,w2.T) 
    loss_0 = np.dot(loss_1,w1.T)

    print('训练第'+str(times)+'次')
    print('loss_out:'+str(sum(sum(loss_out))))
    print('loss_1:'+str(sum(sum(loss_1))))
    print('loss_0:'+str(sum(sum(loss_0))))

    #w2_ = np.dot(act1.T,-loss_out)
    
    for j in range(prediction.shape[1]):

        w2_[:,j] = (np.sum(act1.T*(-0.2*prediction*(1-prediction)*loss_out)[:,j],axis=1))/(prediction.shape[0])
    
    for j in range(act1.shape[1]):
        w1_[:,j] = (np.sum((act0.T*(-0.2*act1*(1-act1)*loss_1)[:,j]),axis=1))/(act1.shape[0])
    
    for j in range(act0.shape[1]):
        w0_[:,j] = (np.sum((X_train.T*(-0.2*act0*(1-act0)*loss_0)[:,j]),axis=1))/(act0.shape[0])
    



    w2 = w2-learningrate*w2_
    w1 = w1-learningrate*w1_
    w0 = w0=learningrate*w0_
    
    #print(w0)
    print('-------')


#-----------测试---------------------------

h0 = np.dot(X_test,w0)
a0 = activation_function(h0)

h1 = np.dot(a0,w1)
a1 = activation_function(h1)

o1 = np.dot(a1,w2)
p = activation_function(o1)


key = 0
for i in range(p.shape[0]):
    if y_test[i] == np.argmax(p[i]):
        key +=1
print('训练'+str(train_times)+'次：'+str(key/p.shape[0]))

#-------------------
np.savetxt(fname="w0.csv", X=w0, fmt="%d",delimiter=",")
np.savetxt(fname="w1.csv", X=w1, fmt="%d",delimiter=",")
np.savetxt(fname="w2.csv", X=w2, fmt="%d",delimiter=",")