import numpy as np
from loss import Loss_CategoricalCrossEntropy



class Activation:

    def forward(self,input):
        pass



class Activation_ReLU(Activation):

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<0] = 0



class Activation_Softmax(Activation):

    def forward(self, inputs):
        exp_input = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        self.output = exp_input/np.sum(exp_input,axis=1,keepdims=True)

    def backward(self,dvalues):
        self.dinputs=np.empty_like(dvalues)
        for index, (single_output,single_dvalue) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian = np.diagflat(single_output) - np.dot(single_output.T,single_output)
            self.dinputs[index] = np.dot(jacobian,single_dvalue)

class Activation_Softmax_Loss_CategoricalCrossEntropy(Activation):

    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs=inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self,dvalues,y_true):
        n_samples = len(dvalues)
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis=0)
        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples),y_true]-=1
        self.dinputs = self.dinputs / n_samples
        
