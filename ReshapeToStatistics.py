#%% Imports
from statistics import variance
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
A = pd.read_pickle(datapath+'TodoAudios_Mel64.pkl')
fs = 48000
print('leido')

#%%
Todo = pd.DataFrame()
for a in A.iD.unique():
    audi = A[A.iD == a]
    for s in audi.slice.unique():
        sli = audi[audi.slice == s]
        mediana = np.median(sli.iloc[:,:64],axis=0)
        media = np.mean(sli.iloc[:,:64],axis=0)
        varianza = np.var(sli.iloc[:,:64],axis=0)
        m = pd.DataFrame([mediana])
        m = m.append(media,ignore_index=True)
        m = m.append(varianza,ignore_index=True)

        aux= pd.concat([m.iloc[0,:],m.iloc[1,:],m.iloc[2,:]],axis=0,ignore_index=True)
        Todo = pd.concat([Todo,aux],axis = 1)