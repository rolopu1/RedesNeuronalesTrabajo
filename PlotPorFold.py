#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% Read Data
datapath = "C:/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "C:/Master/RedesNeuronales/Trabajo/Graficas/Varios/"
A = pd.read_pickle(datapath+'AllSound16000.pkl')

print('leido')
#%% Separate data
for j in np.arange(1,11):
    numClass = []
    fol = A[A['fold']==str(j)]
    for i in range(10):
        numClass.append(fol[fol['dura']==str(i)].shape[0])
    print(numClass)

    fig1, ax1 = plt.subplots()
    ax1.pie(numClass, labels=['0: Air conditioner','1: Car horn','2: Children playing','3: Dogbark','4: Drilling','5: Engine idling','6: Gun shot','7: Jackhammer','8: Siren','9: Street music'], autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig(gaddress+"Clases"+str(j)+".png")

print('Fin')
# %%