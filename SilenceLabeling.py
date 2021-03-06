#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from scipy import signal
import numpy as np
from scipy.fft import fftshift
import librosa
#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/win24000/"
A = pd.read_pickle(datapath+'AllSound.pkl')  #AllSound
fs = 48000
A = A[A['duration']>=0.1]
print('leido')
Todo = pd.DataFrame()
plot = False
#%% Separate data
for i in range(10): 
    Clase = A[A['class']==str(i)]
    print(i)
    for j in Clase['iD'].unique():#['162540']:#
        
        Audio = Clase[Clase['iD']==j]
        Audio = Audio.reset_index(drop=True)
        for k in range(Audio.shape[0]):
            Aux = pd.DataFrame()

            var = 1/np.max(np.abs([np.max(Audio.loc[k,'sound']),np.min(Audio.loc[k,'sound'])]))
            norm = Audio.loc[k,'sound'] * var
            length = 48000
            w = signal.get_window('hamming',length,True)
            Sxx = librosa.feature.melspectrogram(norm,sr= fs, n_fft = length, win_length=length, hop_length=4800, n_mels = 128)

            Datos = pd.DataFrame(Sxx.T)
            Datos['duration'] = Audio.loc[k,'duration']
            Datos['iD'] = Audio.loc[k,'iD']
            Datos['slice'] = Audio.loc[k,'slice']
            Datos['fold'] = Audio.loc[k,'fold']  
            Datos['class'] = Audio.loc[k,'class']  
            lim = np.max(Sxx)/40
            for fi in range(Sxx.shape[1]):
                # print(np.max(Sxx[:,fi]))
                if(np.max(Sxx[:,fi])<lim):
                    Datos.loc[fi,'class'] = 10
            
            Todo = pd.concat([Todo,Datos])
            if plot:
                clases = Datos['class']
                plt.subplot(211)
                plt.plot(np.linspace(0,Audio.loc[k,'duration'],norm.shape[0]),norm)
                plt.ylabel("Amplitud")
                plt.subplot(212)
                plt.pcolormesh(Sxx)#plt.pcolormesh(t, f, Sxx)
                plt.xlabel("time")
                plt.ylabel("Mel coeficient")
                plt.xticks(np.linspace(0,clases.shape[0],clases.shape[0]),clases, rotation = 45)
                plt.savefig(gaddress+"Clase"+str(i)+"iD"+str(j)+"order"+str(k)+".png")
                plt.close()
Todo.to_pickle(datapath+"TodoAudios_Mel12848000.pkl")
print('Fin')
