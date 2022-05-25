#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns

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
A = pd.read_pickle(datapath+'TodoAudios_EstadisticasMel128.pkl')
fs = 48000
print('leido')

#%%

KNNmodel = KNeighborsClassifier(n)
linDis = LinearDiscriminantAnalysis()
Logreg = LogisticRegression(max_iter=1000)
svm = SVC(C=0.1)
gNB = GaussianNB()
RanF = RandomForestClassifier(n_estimators=2, random_state=1)
AdaB = AdaBoostClassifier()

nombres = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
classifiers = [ KNNmodel,linDis,Logreg,svm,AdaB]
labelclass = ['KNN','LinDis', 'LogReg','SVM','AdaB']
# VotingClassifier()
ensemble=VotingClassifier(estimators=classifiers,voting='hard',) #soft utiliza probabilidades
classifiers = [ KNNmodel,linDis,Logreg,svm,gNB,RanF,AdaB,ensemble]
labelclass = ['KNN','LinDis', 'LogReg','SVM','gNB','RanF','AdaB',"Voting"]
#%%
pca = PCA(0.95)
d = dict(zip(range(10),["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]))
#%%Scores
Scores = pd.DataFrame(columns=labelclass)
p=np.zeros([10,8])
a=np.zeros([10,8])
r=np.zeros([10,8])
#%% Separate data
for i in np.arange(1,11):
    Test = A[A['fold']==str(i)]
    Train = A[A['fold']!=str(i)]

    print(i)
    # print(Test.shape)
    # print(Train.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(Train.iloc[:,:128*3])
    # x_train = Train.iloc[:,:128]
    x_train = pca.fit_transform(x_train)
    y_train = Train.loc[:,'class'].astype(int)
    x_test = scaler.transform(Test.iloc[:,:128*3])
    # x_test = Test.iloc[:,:128]
    x_test = pca.transform(x_test)
    y_test = Test.loc[:,'class'].astype(int)

    print(x_test.shape)
    print(x_train.shape)

    for c,cl,k in zip(classifiers,labelclass,range(len(classifiers))):#len(classifiers)
        print(cl)
        c.fit(x_train,y_train)
        pred = c.predict(x_test)
        cm = confusion_matrix(y_test,pred,normalize='true')
        p[i-1,k] = precision_score(y_test,pred,average='macro')
        a[i-1,k] = accuracy_score(y_test,pred)
        r[i-1,k] = recall_score(y_test,pred,average='macro')

        ax = sns.heatmap(np.round(cm,3)*100, annot=True, cmap='Blues',xticklabels=nombres,yticklabels=nombres)
        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('Actual Category ')
        # plt.show()
        plt.savefig(gaddress+"ConfMat_Test_"+str(i)+cl+".png")
        plt.close()
    print(p[i-1,:])

dict
Scores = pd.DataFrame(p)
Scores.to_csv(gaddress+"Precission.csv")
Scores = pd.DataFrame(a)
Scores.to_csv(gaddress+"Accuracy.csv")
Scores = pd.DataFrame(r)
Scores.to_csv(gaddress+"Recall.csv")
Scores = pd.DataFrame([np.mean(p,axis = 0),np.mean(a,axis = 0),np.mean(r,axis = 0)])
Scores.to_csv(gaddress+"Scores.csv")
# %%
