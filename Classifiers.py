#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
from tensorflow.keras.utils import to_categorical

#%%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'TodoAudios_MelSpectrogram.pkl')
fs = 48000
print('leido')

#%%

KNNmodel = KNeighborsClassifier(n_neighbors=10)
linDis = LinearDiscriminantAnalysis()
Logreg = LogisticRegression(max_iter=1000)
svm = SVC()
gNB = GaussianNB()
RanF = RandomForestClassifier(n_estimators=50, random_state=1)
AdaB = AdaBoostClassifier()

classifiers = [ KNNmodel,linDis,Logreg,svm,gNB,RanF,AdaB]
labelclass = ['KNN','LinDis', 'LogReg','SVM','gNB','RanF','AdaB']
# VotingClassifier()

#%%
pca = PCA(0.95)

#%%Scores
Scores = pd.DataFrame(columns=labelclass)
p=list()
a=list()
r=list()
#%% Separate data
for i in np.arange(1,9):
    Test = A[A['fold']==str(i)]
    Train = A[A['fold']!=str(i)]

    print(i)
    print(Test.shape)
    print(Train.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(Train.iloc[:,:128])
    # x_train = Train.iloc[:,:128]
    x_train = pca.fit_transform(x_train)
    y_train = Train.loc[:,'class'].astype(int)
    x_test = scaler.transform(Test.iloc[:,:128])
    # x_test = Test.iloc[:,:128]
    x_test = pca.transform(x_test)
    y_test = Test.loc[:,'class'].astype(int)



    for c,cl in zip(classifiers,labelclass):
        c.fit(x_train,y_train)
        pred = c.predict(x_test)
        cm = confusion_matrix(y_test,pred,normalize='true')
        p.append(precision_score(y_test,pred,average='macro'))
        a.append(accuracy_score(y_test,pred))
        r.append(recall_score(y_test,pred,average='macro'))

    Scores = pd.concat([Scores,pd.DataFrame([p,a,r])])

    ax = sns.heatmap(np.round(cm,3)*100, annot=True, cmap='Blues')
    ax.set_xlabel('Predicted Category')
    ax.set_ylabel('Actual Category ')
    # plt.show()
    plt.savefig(gaddress+"ConfMat_Test_"+str(i)+".png")
    plt.close()



# %%
