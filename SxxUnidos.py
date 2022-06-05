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

            samples =  np.empty((0,Sxx.shape[0],Sxx.shape[1]))

            Datos = pd.DataFrame([Audio.loc[k,'duration'],Audio.loc[k,'iD'],Audio.loc[k,'slice'],Audio.loc[k,'fold'],Audio.loc[k,'class']], columns=['duration','iD','slice','fold','class'])
       
            
            Todo = pd.concat([Todo,Datos])

Todo.to_pickle(datapath+"TodoAudios_Mel12848000.pkl")
print('Fin')
