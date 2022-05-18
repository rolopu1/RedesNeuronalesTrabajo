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
A = pd.read_pickle(datapath+'AllSound.pkl')
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
            w = signal.get_window('hamming',4800,True)
            f, t, Sxx = signal.spectrogram(norm, fs, window = w, nperseg=4800,noverlap=4800/4)

            Datos = pd.DataFrame(Sxx.T)
            Datos['duration'] = Audio.loc[k,'duration']
            Datos['iD'] = Audio.loc[k,'iD']
            Datos['slice'] = Audio.loc[k,'duration']
            Datos['fold'] = Audio.loc[k,'fold']  
            Datos['class'] = Audio.loc[k,'class']  
            lim = np.max(Sxx)/100
            for fi in range(Sxx.shape[1]):
                # print(np.max(Sxx[:,fi]))
                if(np.max(Sxx[:,fi])<lim):
                    Datos.loc[fi,'class'] = 10

            clases = Datos['class']
            
            Todo = pd.concat([Todo,Datos])
            if plot:
                plt.subplot(211)
                
                plt.plot(np.linspace(0,Audio.loc[k,'duration'],norm.shape[0]),norm)
                plt.ylabel("Amplitud")

                # plt.xticks(np.linspace(0,int(Audio.loc[k,'duration']),int(np.ceil(Sxx.shape[0])/div+1),endpoint=True),clases[::div], rotation = 45)
                plt.subplot(212)
                plt.pcolormesh(t, f, Sxx)
                plt.xlabel("time")
                plt.ylabel("Frequency")
                plt.yscale('log')
                plt.ylim([1,24000])
                plt.xticks(np.linspace(0,Audio.loc[k,'duration'],int(np.ceil(Sxx.shape[1])),endpoint=True),clases, rotation = 45)

                plt.savefig(gaddress+"Clase"+str(i)+"iD"+str(j)+"order"+str(k)+".png")

                # plt.show()
                plt.close()
Todo.to_pickle(datapath+"TodoAudios_STFT.pkl")
print('Fin')
