#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/Varios/"
A = pd.read_pickle(datapath+'AllSound.pkl')

print('leido')
#%% Separate data
fig, ax = plt.subplots(2,5)
for c in range(10):
    C = A[A['class']==str(c)]
    
    ax[c//5,c%2].hist(C.duration, bins = 30, color = "blue", rwidth=0.3)
    

plt.savefig(gaddress+"Duracion_ClassJoin"+str(c)+".png")

print('Fin')
# %%