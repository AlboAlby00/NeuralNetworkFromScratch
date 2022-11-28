import unittest
import numpy as np
import loss 

class Test_Loss_CategoricalCrossEntropy(unittest.TestCase):


    def test_forward(self):

        loss_function = loss.Loss_CategoricalCrossEntropy()
        y_pred=np.array([[0.2,0.2,0.6],
                        [0.1,0.1,0.8],
                        [0.0,1.0,0.0]])
        y_true = np.array([2,2,1])
        expected = np.average([-np.log(0.6),-np.log(0.8),-np.log(1-1e-7)])
        calculated = loss_function.calculate(y_pred,y_true)
        self.assertEqual(calculated,expected)

    def test_forward_one_hot(self):

        loss_function = loss.Loss_CategoricalCrossEntropy()
        y_pred=np.array([[0.2,0.2,0.6],
                        [0.1,0.1,0.8],
                        [0.0,1.0,0.0]])
        y_true = np.array([[0,0,1],
                           [0,0,1],
                           [0,1,0]])
        expected = np.average([-np.log(0.6),-np.log(0.8),-np.log(1-1e-7)])
        calculated = loss_function.calculate(y_pred,y_true)
        self.assertEqual(calculated,expected)



class Test_Accuracy(unittest.TestCase):

    def test_accuracy(self):

        y_pred=np.array([[0.6,0.2,0.2],
                        [0.1,0.1,0.8],
                        [0.0,1.0,0.0]])
        y_true = np.array([2,2,1])
        self.assertAlmostEqual(loss.calculate_accuracy(y_pred,y_true),0.66,delta=0.01)

    def test_accuracy_one_hot(self):

        y_pred=np.array([[0.6,0.2,0.2],
                        [0.1,0.1,0.8],
                        [0.0,1.0,0.0]])
        y_true = np.array([[0,0,1],
                           [0,0,1],
                           [0,1,0]])
        self.assertAlmostEqual(loss.calculate_accuracy(y_pred,y_true),0.66,delta=0.01)