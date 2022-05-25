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
scaler = StandardScaler()

Recopilatorio = pd.DataFrame()
for ID in A.iD.unique():
    aud = A[A['iD']==ID]
    for sli in aud.slice.unique():
        
        parte = aud[aud['slice']==sli]
        for j in range(parte.shape[0]-5):
            Recopilatorio = pd.concat([Recopilatorio,pd.concat([pd.DataFrame(parte.iloc[j:j+5,:128].to_numpy().flatten()),parte.iloc[j,129:]],ignore_index = True)],ignore_index = True,axis =1)
RecopilatorioT = Recopilatorio.T
RecopilatorioT.to_pickle(datapath+"RecopilatorioJoinWindow.pkl")