import os

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")   # so the run works headless / in CI
import matplotlib.pyplot as plt

np.random.seed(0)   # reproducible curve

X_t = np.arange(-70, 10, 0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1) + np.exp((0.5*X_t + 20)*0.05)

#plt.plot(X_t, Y_t)



from LSTMcell import *
n_neurons=200
lstm = LSTM(n_neurons)
T = max(X_t.shape)
dense1 = Layer_Dense(n_neurons, T)
dense2 = Layer_Dense(T,1)

lr = float(os.environ.get('LR', 2e-5))
n_epoch = int(os.environ.get('EPOCHS', 800))
Monitor = np.zeros(n_epoch)

for n in range(n_epoch):
    lstm.forward(X_t)
    H = np.array(lstm.H)
    H = H.reshape((H.shape[0], H.shape[1]))

    dense1.forward(H[1:,:])
    dense2.forward(dense1.output)

    Y_hat = dense2.output

    dY = Y_hat - Y_t

    L = (0.5 * np.dot(dY.T, dY) / T).item()   # .item(): NumPy 2 removed float(ndarray)

    Monitor[n] = L

    dense2.backward(dY)
    dense1.backward(dense2.dinputs)

    lstm.backward(dense1.dinputs)

    dense1.weights -= lr*dense1.dweights
    dense2.weights -= lr*dense2.dweights

    dense1.biases -= lr*dense1.dbiases
    dense2.biases -= lr*dense2.dbiases

    lstm.Uf -= lr*lstm.dUf
    lstm.Ui -= lr*lstm.dUi
    lstm.Uo -= lr*lstm.dUo
    lstm.Ug -= lr*lstm.dUg
    
    lstm.Wf -= lr*lstm.dWf
    lstm.Wi -= lr*lstm.dWi
    lstm.Wo -= lr*lstm.dWo
    lstm.Wg -= lr*lstm.dWg

    lstm.bf -= lr*lstm.dbf
    lstm.bi -= lr*lstm.dbi
    lstm.bo -= lr*lstm.dbo
    lstm.bg -= lr*lstm.dbg

    print(f' current MSSE = {L: 0.3f}')

os.makedirs('assets', exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

# Log scale: the first epoch spikes to ~35 and would flatten everything else.
ax1.semilogy(range(n_epoch), Monitor, color='#2a78d6', lw=2)
ax1.set_xlabel('epoch')
ax1.set_ylabel('mean sum of squared error (log)')
ax1.set_title(f'Training loss — {n_epoch} epochs, lr={lr}')
ax1.grid(alpha=0.25, which='both')

ax2.plot(X_t, Y_t, color='#b0b0b0', lw=3, label='target')
ax2.plot(X_t, Y_hat, color='#eb6834', lw=1.5, label='LSTM prediction')
ax2.set_xlabel('t')
ax2.set_ylabel('y')
ax2.set_title(f'Fit after training — LSTM({n_neurons}) + 2 dense layers')
ax2.legend(frameon=False)
ax2.grid(alpha=0.25)

fig.suptitle('Pure NumPy LSTM, forward and backward written by hand',
             fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig('assets/loss_curve.png', dpi=160)
print(f'first epoch MSSE = {Monitor[0]:.3f}, last = {Monitor[-1]:.3f}')
print('wrote assets/loss_curve.png')
if os.environ.get('DISPLAY'):
    plt.show()

 















