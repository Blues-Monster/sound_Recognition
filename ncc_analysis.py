#模型训练情况分析
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



#--------特征规范化----------
#提取数据文件
music_dict = {'blues':0,'classical':1,'country':2,'disco':3,'hiphop':4,'jazz':5,'metal':6,'pop':7,'reggae':8,'rock':9}
data = pd.read_csv('./music_data/data.csv',encoding= 'utf-8')
label = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
y =[music_dict[i] if i in music_dict else i for i in label]
#对矩阵进行归一化
trans_X = [0.00001,100000,100,100000,1000,0.01,0.001,0.00001,0.001,0.000001,0.01,0.01]+[0.1]*3+[1]+[0.1]+[1]+[0.1]+[1]*2+[1]*9+[0.001]+[0.01]*20+[100]*23+[10]+[0.01]
#print(len(trans_X))
X = np.dot(X,np.diag(trans_X))
#拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=314)

#--------激活函数---------------

def activation_function(x):
    #if(x>0):
    return 1/(1+np.exp(-0.1*x))
    #else:
    #    return 1/(1+np.exp(0.01*x))
    #return x



#-----------标准输出矩阵------------
oneHot = np.identity(len(music_dict))
y_true = oneHot[y_train]


#------------生成神经网络中间层--------------

for k in range(6):

    w0 = pd.read_csv('w0_'+str(200*k+200)+'times.csv',encoding= 'utf-8',header=None)
    w1 = pd.read_csv('w1_'+str(200*k+200)+'times.csv',encoding= 'utf-8',header=None)
    w2 = pd.read_csv('w2_'+str(200*k+200)+'times.csv',encoding= 'utf-8',header=None)


    h0 = np.dot(X_train,w0)
    a0 = activation_function(h0)

    h1 = np.dot(a0,w1)
    a1 = activation_function(h1)

    o1 = np.dot(a1,w2)
    p_train = activation_function(o1)
    
    loss = sum(sum(np.square(p_train-y_true)))


    h0 = np.dot(X_test,w0)
    a0 = activation_function(h0)

    h1 = np.dot(a0,w1)
    a1 = activation_function(h1)

    o1 = np.dot(a1,w2)
    p_test = activation_function(o1)


    key_train = 0
    for i in range(p_train.shape[0]):
        if y_train[i] == np.argmax(p_train[i]):
            key_train +=1
    key_test = 0
    for i in range(p_test.shape[0]):
        if y_test[i] == np.argmax(p_test[i]):
            key_test +=1
    
    print('train:'+str(k*200+200)+'times')
    print(key_train/p_train.shape[0])
    print('test:'+str(k*200+200)+'times')
    print(key_test/p_test.shape[0])
    print('loss:'+str(loss))
    print('---------------------------')