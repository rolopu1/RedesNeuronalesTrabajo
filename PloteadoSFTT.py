#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'Sound1.pkl')

print('leido')
#%% Separate data
numClass = []
for i in range(10):
    Clase = A[A['class']==str(i)]
    print(i)
    for j in Clase['iD'].unique():
        Audio = Clase[Clase['iD']==j]
        Audio = Audio.reset_index(drop=True)
        for k in range(Audio.shape[0]):
            f,t,Zxx = signal.stft(Audio.loc[k,'sound'], 48000, nperseg=1024)

            plt.specgram(Audio.loc[k,'sound'],Fs=48000, cmap='nipy_spectral')
            plt.savefig(gaddress+"Clase"+str(i)+"iD"+str(j)+"order"+str(k)+".png")
            plt.close()
print('Fin')
# %%
