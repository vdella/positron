###############
##### Plotando fronteira de decisão não-linear
###############

import numpy as np
import matplotlib.pyplot as plt

# Plotando fronteira de decisão
x1s = np.linspace(-1,1.5,50)
x2s = np.linspace(-1,1.5,50)
z=np.zeros((len(x1s),len(x2s)))

#y = h(x) = 1/(1+exp(- z))
#z = theta.T * x

for i in range(len(x1s)):
    for j in range(len(x2s)):
        x = np.array([x1s[i], x2s[j]]).reshape(2,-1)
        # z[i,j] = net_z_output( x )  # saida do modelo antes de aplicar a função sigmoide - substituir aqui teu código
plt.contour(x1s,x2s,z.T,0)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc=0)


###############
##### Classificação binária com modelo de rede neural - backprop / regressão logística
###############