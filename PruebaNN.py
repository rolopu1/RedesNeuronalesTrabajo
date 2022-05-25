#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns

#%%
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.optimizers

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'TodoAudios_Mel128.pkl')
fs = 48000
print('leido')

#%%


#%%
buenos = np.zeros((10,10))
total = np.zeros((10,10))
predicciones = np.zeros((10,10))
for i in np.arange(1,11):
    model = Sequential()
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(i)

    Test = A[A['fold']==str(i)]
    Train = A[A['fold']!=str(i)]
    scaler = StandardScaler()
    Train.iloc[:,:128] = scaler.fit_transform(Train.iloc[:,:128])
    Recopilatorio = pd.DataFrame()
    for ID in Train.iD.unique():
        aud = Train[Train['iD']==ID]
        for sli in aud.slice.unique():
            
            parte = aud[aud['slice']==sli]
            for j in range(parte.shape[0]-5):
                Recopilatorio = pd.concat([Recopilatorio,pd.concat([pd.DataFrame(parte.iloc[j:j+5,:128].to_numpy().flatten()),parte.iloc[j,129:]],ignore_index = True)],ignore_index = True,axis =1)
    RecopilatorioT = Recopilatorio.T
    x_train = RecopilatorioT.iloc[:,:128]
    y_train = parte.loc[:,'class'].astype(int)
            # print(x_train.shape)
    model.fit(x_train, y_train,epochs = 1, verbose=1,batch_size=4096)
    # for a in Test

    for ID in Test.iD.unique():
        aud = Test[Test['iD']==ID]
        for sli in aud.slice.unique():
            
            parte = aud[aud['slice']==sli]
            x_test = scaler.transform(parte.iloc[:,:128])

            clases = parte[parte['class']!=10]
            y_true_audio = int(clases.iloc[-1,-1])

            prediccion = model.predict(x_test)
            sumas = np.sum(prediccion[:,:10],axis=0)
            pred = np.where(max(sumas) == sumas)

            predicciones[i-1,pred[0]] = predicciones[i-1,pred[0]] +1 
            total[i-1,y_true_audio] = total[i-1,y_true_audio] +1 

            if (pred[0] == y_true_audio)[0]:
                buenos[i-1,y_true_audio] = buenos[i-1,y_true_audio] +1
    print(buenos)
print(np.mean(buenos/total,axis = 0))
print(buenos)
print(total)
# %%
