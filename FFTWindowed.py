#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from sklearn.decomposition import PCA

#%% Read Data
datapath = "E:/RODRIGO/UNI/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "E:/RODRIGO/UNI/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'Sound1.pkl')
fs=48000
print('leido')

#%% Welch method
All = pd.DataFrame()
pca = PCA(0.95)
for i in range(A.shape[0]):
    freq, time, complejo = signal.stft(A.loc[i,'sound'], fs, nperseg=1024)
    
    Resultado = pd.DataFrame(np.Abs(complejo))
    All = All.append(Resultado)
# %%
