#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import librosa

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'AllSound.pkl')

print('leido')
#%% Separate data
for j in A['iD'].unique():
    Audio = A[A['iD']==j]
    Audio = Audio.reset_index(drop=True)
    for k in range(Audio.shape[0]):
        s = Audio.loc[k,'sound']
        mfccs = librosa.feature.mfcc(y=s, sr=48000, n_mfcc = 40)

        plt.savefig(gaddress+'MFCCs/'+str(j)+'_'+str(k)+'.png')
# %%
