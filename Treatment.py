#%% Imports
import pandas as pd
import matplotlib.pyplot as plt

#%% Read Data
datapath = "E:/RODRIGO/UNI/Master/RedesNeuronales/Trabajo/UrbanSound8K/audio/"
gaddress = "E:/RODRIGO/UNI/Master/RedesNeuronales/Trabajo/Graficas/"
A = pd.read_pickle(datapath+'AllSound')

print('leido')
#%% Separate data
numClass = []
for i in range(10):
    numClass.append(A[A['class']==str(i)].shape[0])
print(numClass)

fig1, ax1 = plt.subplots()
ax1.pie(numClass, labels=['0: Air conditioner','1: Car horn','2: Children playing','3: Dogbark','4: Drilling','5: Engine idling','6: Gun shot','7: Jackhammer','8: Siren','9: Street music'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig(gaddress+"Clases.png")

print('Fin')
# %%
