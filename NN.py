#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
import random
import tensorflow as tf
#%%
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
import keras.optimizers
# from plot_keras_history import show_history, plot_history
#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'TodoAudios_Mel12848000.pkl')
fs = 48000
print('leido')

#%%
buenos = np.zeros((10,10))
total = np.zeros((10,10))
predicciones = np.zeros((10,10))
cm = np.zeros((10,10))
for i in np.arange(1,11):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))

    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Dense(128))
    # model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(11))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    Test = A[A['fold']==str(i)]
    Train = A[A['fold']!=str(i)]

    print(i)

    scaler = StandardScaler()
    # x_train = scaler.fit_transform(Train.iloc[:,:128])
    x_train = Train.iloc[:,:128]
    y_train = Train.loc[:,'class'].astype(int)
    # x_test = scaler.transform(Test.iloc[:,:128])
    x_test = Test.iloc[:,:128]
    y_test = Test.loc[:,'class'].astype(int)
    batch = 2048
    # x_train = tf.random.shuffle(x_train)
    # 

    history = model.fit(x_train, y_train,epochs = 1, verbose=1,batch_size=batch, validation_data=(x_test,y_test))
    # for a in Test
    print(model.summary())
    for ID in Test.iD.unique():
        aud = Test[Test['iD']==ID]
        for sli in aud.slice.unique():
            
            parte = aud[aud['slice']==sli]
            # x_test = scaler.transform(parte.iloc[:,:128])
            x_test = parte.iloc[:,:128]
            clases = parte[parte['class']!=10]
            y_true_audio = int(clases.iloc[-1,-1])

            prediccion = model.predict(x_test)

            pred = np.where(np.max(prediccion[:,:10])==prediccion[:,:10])[1]
            pred = pred[0]
            predicciones[i-1,pred] = predicciones[i-1,pred] +1 
            total[i-1,y_true_audio] = total[i-1,y_true_audio] +1 
            cm[y_true_audio, pred] = cm[y_true_audio, pred] +1
            if (pred == y_true_audio):
                buenos[i-1,y_true_audio] = buenos[i-1,y_true_audio] +1
    print(np.sum(buenos/total,axis = 1)*10)
    print(buenos[i-1,:])
    print(predicciones[i-1,:])
    print(buenos[i-1,:]/total[i-1,:])
print(model.summary())
print(np.mean(np.sum(buenos/total,axis = 1)*10))
d = dict(zip(range(10),["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]))
nombres = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
ax = sns.heatmap(np.round(cm,3)/10, annot=True, cmap='Blues',xticklabels=nombres,yticklabels=nombres)
plt.tight_layout()
plt.savefig(gaddress+"ConfMat_Test_"+str(i)+"NN.png")
plt.close()
print("fin")
