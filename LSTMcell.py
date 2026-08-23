import numpy as np
from utils import Sigmoid,TanH

class LSTM():

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons

        #forget gate
        self.Uf = np.zeros((n_neurons, 1 ))
        self.bf = np.zeros((n_neurons, 1 ))
        self.Wf = np.zeros((n_neurons, n_neurons))

        #Input gate
        self.Ui = np.zeros((n_neurons, 1 ))
        self.bi = np.zeros((n_neurons, 1 ))
        self.Wi = np.zeros((n_neurons, n_neurons))

        #Output gate
        self.Uo = np.zeros((n_neurons, 1 ))
        self.bo = np.zeros((n_neurons, 1 ))
        self.Wo = np.zeros((n_neurons, n_neurons))

        #c ~ 
        self.Ug = np.zeros((n_neurons, 1 ))
        self.bg = np.zeros((n_neurons, 1 ))
        self.Wg = np.zeros((n_neurons, n_neurons))

    def forward(self, X_t):

        T = max(X_t.shape)

        self.T = T
        self.X_t = X_t

        n_neurons = self.n_neurons

        self.H = [np.zeros((n_neurons, 1)) for t in range(T+1)]
        self.C = [np.zeros((n_neurons, 1)) for t in range(T+1)]
        self.C_tilde = [np.zeros((n_neurons, 1)) for t in range(T)]

        self.F = [np.zeros((n_neurons, 1)) for t in range(T)]
        self.O = [np.zeros((n_neurons, 1)) for t in range(T)]
        self.I = [np.zeros((n_neurons, 1)) for t in range(T)]


        #forget gate
        self.dUf = np.zeros((n_neurons, 1 ))
        self.dbf = np.zeros((n_neurons, 1 ))
        self.dWf = np.zeros((n_neurons, n_neurons))

        #Input gate
        self.dUi = np.zeros((n_neurons, 1 ))
        self.dbi = np.zeros((n_neurons, 1 ))
        self.dWi = np.zeros((n_neurons, n_neurons))

        #Output gate
        self.dUo = np.zeros((n_neurons, 1 ))
        self.dbo = np.zeros((n_neurons, 1 ))
        self.dWo = np.zeros((n_neurons, n_neurons))

        #c ~ 
        self.dUg = np.zeros((n_neurons, 1 ))
        self.dbg = np.zeros((n_neurons, 1 ))
        self.dWg = np.zeros((n_neurons, n_neurons))


        SigmF = [Sigmoid() for t in range(T)]
        SigmI = [Sigmoid() for t in range(T)]
        SigmO = [Sigmoid() for t in range(T)]
        
        Tanh1 = [TanH() for t in range(T)]
        Tanh2 = [TanH() for t in range(T)]

        ht = self.H[0]
        ct = self.C[0]

        #calling the LSTM cell:
        [H, C, SigmF, SigmI, SigmO, Tanh1, Tanh2, F, O, I, C_tilde]=self.LSTMCell(X_t, ht, ct, SigmF, SigmI, SigmO, Tanh1, Tanh2, 
                 self.H, self.C, self.F, self.O, self.I, self.C_tilde) 

        self.F = F
        self.O = O
        self.I = I
        self.C_tilde = C_tilde

        self.H = H
        self.C = C


        self.sigmF = SigmF
        self.sigmI = SigmI
        self.sigmO = SigmO
        self.Tanh1 = Tanh1
        self.Tanh2 = Tanh2
 
    def LSTMCell(self, X_t, ht, ct, SigmF, SigmI, SigmO, Tanh1, Tanh2, 
                 H, C, F, O, I, C_tilde):
        
        for t, xt in enumerate(X_t):
            
            xt = xt.reshape(1,1)
            
            #forget gate
            outf = np.dot(self.Uf, xt) + np.dot(self.Wf, ht) + self.bf
            SigmF[t].forward(outf)
            ft = SigmF[t].output

            #input gate
            outi = np.dot(self.Ui, xt) + np.dot(self.Wi, ht) + self.bi
            SigmI[t].forward(outi)
            it = SigmI[t].output

            #output gate
            outo = np.dot(self.Uo, xt) + np.dot(self.Wo, ht) + self.bo
            SigmO[t].forward(outo)
            ot = SigmO[t].output


            #c tilde gate
            outct_tilde = np.dot(self.Ug,xt) + np.dot(self.Wg, ht) + self.bg
            Tanh1[t].forward(outct_tilde)
            ct_tilde = Tanh1[t].output

            # c(t) = f(t) x c(t-1) + i(t)xc~(t)
            ct = np.multiply(ft, ct) + np.multiply(it, ct_tilde)

            Tanh2[t].forward(ct)
            ht = np.multiply(Tanh2[t].output, ot)

            H[t+1] = ht
            C[t+1] = ct
            C_tilde[t] = ct_tilde

            F[t] = ft
            O[t] = ot
            I[t] = it
        
        return(H, C, SigmF, SigmI, SigmO, Tanh1, Tanh2, F, O, I, C_tilde)



    def backward(self,dvalues):
        
        #dh  = inputs from the dense layer

        T = self.T
        H = self.H
        C = self.C

        O = self.O
        I = self.I
        C_tilde = self.C_tilde

        X_t = self.X_t

        sigmF = self.sigmF
        sigmI = self.sigmI
        sigmo = self.sigmO
        Tanh1 = self.Tanh1
        Tanh2 = self.Tanh2

        dht = dvalues[-1,:].reshape(self.n_neurons, 1)

        #Back propagation:
        for t in reversed(range(T)):

            xt = X_t[t].reshape(1,1)

            Tanh2[t].backward(dht)
            dtanh2 = Tanh2[t].dinputs

            dhtdtanh = np.multiply(O[t], dtanh2)

            dctdft = np.multiply(dhtdtanh, C[t-1])
            dctdit = np.multiply(dhtdtanh, C_tilde[t])
            dctdct_tilde = np.multiply(dhtdtanh, I[t])

            Tanh1[t].backward(dctdct_tilde)
            dtanh1 = Tanh1[t].dinputs

            sigmF[t].backward(dctdft)
            dsigmf = sigmF[t].dinputs

            sigmI[t].backward(dctdit)
            dsigmi = sigmI[t].dinputs

            sigmo[t].backward(np.multiply(dht, Tanh2[t].output))
            dsigmo = sigmo[t].dinputs
            
            
            dsigmfdUf = np.dot(dsigmf,xt)
            dsigmfdWf  = np.dot(dsigmf, H[t-1].T)

            self.dUf += dsigmfdUf
            self.dWf += dsigmfdWf
            self.dbf += dsigmf
            

            dsigmfdUi = np.dot(dsigmi,xt)
            dsigmfdWi  = np.dot(dsigmi, H[t-1].T)

            self.dUi += dsigmfdUi
            self.dWi += dsigmfdWi
            self.dbi += dsigmi


            dsigmfdUo = np.dot(dsigmo,xt)
            dsigmfdWo  = np.dot(dsigmo, H[t-1].T)

            self.dUo += dsigmfdUo
            self.dWo += dsigmfdWo
            self.dbo += dsigmo


            dtanh1dUg = np.dot(dtanh1, xt)
            dtanh1dWg = np.dot(dtanh1, H[t-1].T)

            self.dUg += dtanh1dUg
            self.dWg += dtanh1dWg
            self.dbg += dtanh1



            dht = np.dot(self.Wf, dsigmf) + np.dot(self.Wi, dsigmi) +\
                np.dot(self.Wo, dsigmo) + np.dot(self.Wg, dtanh1) +\
                dvalues[t-1, :].reshape(self.n_neurons, 1)


        self.H = H




class Layer_Dense():

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs = np.dot(dvalues, self.weights.T)
