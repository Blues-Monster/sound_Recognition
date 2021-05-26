import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#读取文件获得特征矩阵
data_path = './music_data/music.csv'
data = pd.read_csv(data_path,header=None,encoding= 'utf-8')
music_dict = {'blues':0,'classical':1,'country':2,'disco':3,'hiphop':4,'jazz':5,'metal':6,'pop':7,'reggae':8,'rock':9}
#添加可有可无的特征名称
tip0 = ['Zero Crossing Rate','y_harm_mean','y_harm_var','y_perc_mean','y_perc_var','Tempo BMP','spectral_centroids_mean','spectral_centroids_var','spectral_rolloff_mean','spectral_rolloff_var']
tip1 = ['mfccs_mean']*20
tip2 = ['mfccs_var']*20
tip3 = ['chromagram_mean']*12
tip4 = ['chromagram_var']*12
tip5 = ['chroma_stft_mean','chroma_stft_var','label']
tip = tip0+tip1+tip2+tip3+tip4+tip5

#da = pd.DataFrame(data=data)
data.columns = tip



#分离结果和特征
y = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
y =[music_dict[i] if i in music_dict else i for i in y]

#对矩阵进行归一化
trans_X = [0.00001,100000,100,100000,1000,0.01,0.001,0.00001,0.001,0.000001,0.01,0.01]+[0.1]*3+[1]+[0.1]+[1]+[0.1]+[1]*2+[1]*9+[0.01]+[0.1]*20+[100]*24+[0.1]
X = np.dot(X,np.diag(trans_X))


#拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=314)





# 创建神经网络模型
mlp = MLPClassifier(solver='lbfgs',activation='identity',hidden_layer_sizes = [306, 153],max_iter = 8000)

# 填充数据并训练
mlp.fit(X_train, y_train)
# 评估模型
score = mlp.score(X_test, y_test)
print(score)

