import librosa
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile

datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
Data = pd.DataFrame(columns={'sound','duration','class','slice','fold','iD'})


def obtener_audio():
    ruta = ruta[1:-1] # para eliminar los paréntesis del nombre
    x, sample_rate = librosa.load(os.path.join(folder,file), sr=48000)
    return pd.Series(x for x in x)

for i in np.arange(1,11):
    folder = os.path.join(datapath,"fold"+str(i))

    print(i)
    for file in os.listdir(folder):
        aux = pd.DataFrame()
        if file.endswith('.wav'):
            w= librosa.load(os.path.join(folder,file),sr=48000)

            aux['sound'] = w
            aux['duration'] = w[0].shape[0]/48000
            aux['class'] = file[file.find('-')+1:file.find('-')+2]
            aux['slice'] = file[file.find('-')+5:-4]
            aux['fold'] = str(i)
            aux['iD'] = file[:file.find('-')]

            Data = pd.concat([Data,aux])
            
    Data = Data[Data.index ==0]
    # if i == 1 or i==2 or i==3:
        # Data = Data.reset_index(drop=True)
        #Data.to_pickle(datapath+'/Sound'+str(i)+'.pkl')

Data = Data.reset_index(drop=True)
Data.to_pickle(datapath+'/AllSound.pkl')