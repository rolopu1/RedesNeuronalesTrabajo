#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'AllSound.pkl')

print('leido')
#%% Separate data

print(np.max((A['duration']))) #Máximo de 16 segundos
Longitud = 16 * 48000
#%%
for j in A['iD'].unique():
    Audio = A[A['iD']==j]
    Audio = Audio.reset_index(drop=True)
    for k in range(Audio.shape[0]):
        s = Audio.loc[k,'sound'].shape[0]
        
print('Fin')
