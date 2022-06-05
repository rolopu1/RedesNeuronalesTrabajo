#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
from sklearn.model_selection import train_test_split
#%%
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.optimizers

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'TodoAudios_Mel12848000.pkl')
fs = 48000
print('leido')

#%%

model = Sequential()
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1280))
model.add(Activation('relu')) ## 2 epoch
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(11))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

print(A.shape)

for i in np.arange(1,10):
    Eval = A[A['fold']==str(i)]
    TrTe = A[A['fold']!=str(i)]

    X = TrTe.iloc[:,:128]
    Y = TrTe.iloc[:,-1].astype(int)
    X_eval = Eval.iloc[:,:128]
    Y_eval = Eval.iloc[:,-1].astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.0001, random_state=42)
    # X_train = X
    # y_train = Y

    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)
    X_eval = scaler.transform(X_eval)
    batch = 1024
    model.fit(x_train, y_train,epochs =10, verbose=1,batch_size=batch, validation_batch_size=batch,validation_data=(X_test,y_test))


    pred = model.predict(X_eval)
    pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(Y_eval,pred,normalize='true')
    p= precision_score(Y_eval,pred,average='macro')
    a = accuracy_score(Y_eval,pred)
    r= recall_score(Y_eval,pred,average='macro')
    print(p)
    print(a)
    print(r)
    ax = sns.heatmap(np.round(cm,3)*100, annot=True, cmap='Blues')#,xticklabels=nombres,yticklabels=nombres
    ax.set_xlabel('Predicted Category')
    ax.set_ylabel('Actual Category ')
    # plt.show()
    plt.savefig(gaddress+"ConfMat_Eval_"+str(i)+".png")
    plt.close()
print("fin")