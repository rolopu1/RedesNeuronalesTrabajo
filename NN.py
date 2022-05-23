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
# pca = PCA()
le = LabelEncoder()
score = list()
model = Sequential()
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(11))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

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

    Test = A[A['fold']==str(i)]
    Train = A[A['fold']!=str(i)]

    print(i)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(Train.iloc[:,:128])
    # x_train = Train.iloc[:,:128]
    # x_train = pca.fit_transform(x_train)
    y_train = Train.loc[:,'class'].astype(int)
    
    # x_test = Test.iloc[:,:128]
    # x_test = pca.transform(x_test)

    print(x_train.shape)

    # y_train = le.fit_transform(y_train)
    # y_test = le.fit_transform(y_test)
    batch = 1024
    model.fit(x_train, y_train,epochs = 4, verbose=1,batch_size=batch, validation_batch_size=batch)
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

            #TODO IF prediccion es mas probable la clase silencio, coger la siguiente clase
            if (pred[0] == y_true_audio)[0]:
                buenos[i-1,y_true_audio] = buenos[i-1,y_true_audio] +1
print(np.sum(buenos/total,axis = 1))
# %%
