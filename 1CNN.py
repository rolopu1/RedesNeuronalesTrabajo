#%% Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import tensorflow as tf
#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, MaxPool1D
import keras.regularizers as regularizers
from keras.layers import Input, Dense
from keras.models import Model

def definitionMod():
    inp = Input(shape=(50099,1))

    # ----------------------
    conv1 = Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', padding='valid')(inp)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D(pool_size=8, strides=8)(norm1)
    # ----------------------

    conv2 = Conv1D(filters=32, kernel_size=32, strides=2, activation='relu', padding='valid')(pool1)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPool1D(pool_size=8, strides=8)(norm2)

    # ----------------------
    conv3 = Conv1D(filters=64, kernel_size=16, strides=2, activation='relu', padding='valid')(pool2)
    norm3 = BatchNormalization()(conv3)

    # ----------------------
    conv4 = Conv1D(filters=128, kernel_size=8, strides=2, activation='relu', padding='valid')(norm3)
    norm4 = BatchNormalization()(conv4)

    # ----------------------
    conv5 = Conv1D(filters=256, kernel_size=4, strides=2, activation='relu', padding='valid')(norm4)
    norm5 = BatchNormalization()(conv5)
    pool3 = MaxPool1D(pool_size=4, strides=4)(norm5)

    flat = Flatten()(pool3)
    dense1 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02))(flat)
    drop2 = Dropout(.5)(dense1)
    dense2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(drop2)
    drop2 = Dropout(.25)(dense2)
    dense3 = Dense(10, activation=None)(drop2)
    model = Model(inputs=[inp], outputs=dense3)

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam'
                    , metrics=['accuracy'])

    return model 

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'TodoAudios_Mel12848000.pkl')#AllSoundNorm
fs = 48000
print('leido')

#%%
buenos = np.zeros((10,10))
total = np.zeros((10,10))
predicciones = np.zeros((10,10))
for i in np.arange(1,11):

    num_classes = 11

    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=5,activation='relu', input_shape=(128,1)))
    model.add(MaxPooling1D(pool_size=5 ))
    # model.add(Conv1D(filters=128, kernel_size=6, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2 ))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # model = definitionMod()
    Test = A[A['fold']==str(i)]
    Train = A[A['fold']!=str(i)]

    # print(i)
    x_train = Train.iloc[:,:128]
    y_train = Train.iloc[:,-1]
    x_test = Test.iloc[:,:128]
    y_test = Test.iloc[:,-1]

    model.fit(x_train, y_train.iloc[:].astype(int), epochs=10, validation_data=(x_test, y_test.iloc[:].astype(int)))

