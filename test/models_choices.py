import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
'''
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance
'''

data_path = './music_data/music.csv'
data = pd.read_csv(data_path,header=None)

tip0 = ['过零率','y_harm_mean谐波均值','y_harm_var谐波方差','y_perc_mean感知激波均值','y_perc_var感知激波方差','Tempo BMP','spectral_centroids_mean光谱质心均值','spectral_centroids_var光谱质心方差','spectral_rolloff_mean光谱衰减均值','spectral_rolloff_var光谱衰减方差']
tip1 = ['梅尔频率倒谱系数)均值']*20
tip2 = ['梅尔频率倒谱系数)方差']*20
tip3 = ['chromagram_mean色度频率均值']*12
tip4 = ['chromagram_var色度频率方差']*12
tip5 = ['chroma_stft_mean频率均值','chroma_stft_var频率方差','label']
tip = tip0+tip1+tip2+tip3+tip4+tip5

#da = pd.DataFrame(data=data)
data.columns = tip

y = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
#### NORMALIZE X ####
'''
# Normalize so everything is on the same scale. 

cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

# new data frame with the new scaled data. 

X = pd.DataFrame(np_scaled, columns = cols)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



def model_assess(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')

# Naive Bayes
nb = GaussianNB()
model_assess(nb, "Naive Bayes")

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, "Stochastic Gradient Descent")

# KNN
knn = KNeighborsClassifier(n_neighbors=19)
model_assess(knn, "KNN")

# Decission trees
tree = DecisionTreeClassifier()
model_assess(tree, "Decission trees")

# Random Forest
rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(rforest, "Random Forest")

# Support Vector Machine
svm = SVC(decision_function_shape="ovo")
model_assess(svm, "Support Vector Machine")

# Logistic Regression
lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_assess(lg, "Logistic Regression")

# Neural Nets
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
model_assess(nn, "Neural Nets")
'''
# Cross Gradient Booster
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, "Cross Gradient Booster")

# Cross Gradient Booster (Random Forest)
xgbrf = XGBRFClassifier(objective= 'multi:softmax')
model_assess(xgbrf, "Cross Gradient Booster (Random Forest)")



# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)

print('Accuracy', ':', round(accuracy_score(y_test, preds), 5), '\n')
'''
'''
# Confusion Matrix
confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'
plt.figure(figsize = (16, 9))
sns.heatmap(confusion_matr, cmap="Blues", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
plt.savefig("conf matrix")
plt.show()
'''