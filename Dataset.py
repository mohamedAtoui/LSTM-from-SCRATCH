import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-70, 10, 0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1) + np.exp((0.5*X_t + 20)*0.05)

#plt.plot(X_t, Y_t)



from LSTMcell import *

lstm = LSTM(n_neurons=200)
lstm.forward(X_t)

for c in lstm.C:
    plt.plot(np.arange(20), c[0:20], 'k-', linewidth = 1, alpha = 0.05)

plt.show()