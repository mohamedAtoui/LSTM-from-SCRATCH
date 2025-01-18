import numpy as np

class TanH:

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        

    def backward(self, dvalues):
        deriv = 1 - self.output**2
        self.dinputs = np.multiply(deriv, dvalues)


class Sigmoid:

    def forward(self, M):

        sigm = np.clip(1/(1+np.exp(-M)),1e-7,1-1e-7)
        self.output = sigm
        self.inputs = sigm

    def backward(self, dvalues):
        sigm = self.inputs
        deriv = np.multiply(sigm, (1 - sigm))
        self.dinputs = np.multiply(deriv, dvalues)
