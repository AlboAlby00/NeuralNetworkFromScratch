import unittest
import numpy as np
import activation as f

class Test_ReLU(unittest.TestCase):

    def test_forward(self):
        X=np.array([10,-3,6,2,-8])
        y=np.array([10,0,6,2,0])
        relu = f.Activation_ReLU()
        relu.forward(X)
        self.assertTrue((relu.output == y).all())

    def test_backward(self):
        #TODO
        X=np.array([10,-3,6,2,-8])
        y=np.array([10,0,6,2,0])
        relu = f.Activation_ReLU()
        # relu.backward(X)
        # self.assertTrue((relu.dinputs == y).all())

class Test_Softmax(unittest.TestCase):

    def test_forward(self):
        X=[[0.5,1,1.5]]
        y=[[0.18632372, 0.30719589, 0.50648039]]
        softmax = f.Activation_Softmax()
        softmax.forward(X)
        np.testing.assert_almost_equal(softmax.output, y, decimal=3)

    #TODO
    def test_backward(self):

        softmax_output=np.array([0.7,0.1,0.2]).reshape(-1,1)
        dvalues = np.array([[1,0,0]])
        softmax = f.Activation_Softmax()
        softmax.output = softmax_output
        softmax.backward(dvalues=dvalues)
        expected_dinputs = np.array([0.21,-0.07,-0.14]).reshape(-1,1)
        np.testing.assert_almost_equal(softmax.dinputs,expected_dinputs,decimal=1)
