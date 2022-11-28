import numpy as np

class Loss:

    def calculate(self,predictions,y):
        sample_losses = self.forward(predictions,y)
        data_loss = np.average(sample_losses)
        return data_loss

    def forward(self,y_pred,y_true):
        pass


class Loss_CategoricalCrossEntropy(Loss):


    def forward(self,y_pred,y_true):

        number_samples=len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if(len(y_true.shape)==1):
            prob_correct_class = y_pred_clipped[range(number_samples),y_true]      
        elif(len(y_true.shape)==2):
            prob_correct_class = np.sum(y_pred_clipped*y_true,axis=1)

        error = - np.log(prob_correct_class)
        return error


    def backward(self,dvalues,y_true):

        n_samples = len(dvalues)
        n_classes = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(n_classes)[y_true]

        dinputs = - y_true / dvalues
        self.dinputs = dinputs / n_samples







def calculate_accuracy(y_pred,y_true):

    number_samples=len(y_pred)

    if(len(y_true.shape)==1):
        number_correct_predictions = np.sum(np.argmax(y_pred,axis=0)==y_true)      
    elif(len(y_true.shape)==2):
        number_correct_predictions = np.sum(y_true[range(number_samples),np.argmax(y_pred,axis=0)])
    accuracy = number_correct_predictions/number_samples
    
    return accuracy