import numpy as np
import nnfs
from create_data import spiral_data
from activation import Activation_ReLU
from activation import Activation_Softmax
from layers import Layer_Dense
from loss import Loss_CategoricalCrossEntropy

nnfs.init()

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]



X,y = spiral_data(points=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = Loss_CategoricalCrossEntropy()

dvalues = np.array([[0.2,0.2,0.6],
                    [0.2,0.2,0.6]])

print(dvalues.shape)







