#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from scipy import signal
import numpy as np
from scipy.fft import fftshift
#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'Sound1.pkl')
fs = 48000
A = A[A['duration']>=0.1]
print('leido')
Todo = pd.DataFrame(columns=A.columns)
#%% Separate data
numClass = []
for i in range(10):
    Clase = A[A['class']==str(i)]
    print(i)
    for j in Clase['iD'].unique():
        Audio = Clase[Clase['iD']==j]
        Audio = Audio.reset_index(drop=True)
        for k in range(Audio.shape[0]):

            var = 1/np.max(np.abs([np.max(Audio.loc[k,'sound']),np.min(Audio.loc[k,'sound'])]))
            norm = Audio.loc[k,'sound'] * var
            w = signal.get_window('hamming',2400,True)
            f, t, Sxx = signal.spectrogram(norm, fs, window = w)
            
            plt.subplot(211)
            plt.plot(np.linspace(0,Audio.loc[k,'duration'],norm.shape[0]),norm)
            plt.xlabel("Sample")
            plt.ylabel("Amplitud")
            
            plt.subplot(212)
            plt.pcolormesh(t, f, Sxx)
            plt.xlabel("time")
            plt.ylabel(str(np.max(Sxx)))
            plt.colorbar()

            plt.savefig(gaddress+"Clase"+str(i)+"iD"+str(j)+"order"+str(k)+".png")
            plt.close()


print('Fin')
