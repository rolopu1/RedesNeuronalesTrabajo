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
fig1, ax1 = plt.subplots()
plt.hist(A.duration, bins = 30, color = "blue", rwidth=0.3)

plt.savefig(gaddress+"Duracion.png")

print('Fin')
# %%