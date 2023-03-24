import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self , learning_rate = 0.001 , n_iters = 1000):
        self.lr = learning_rate 
        self.n_iters = n_iters
        self.weight = None
        self.bias = None 

    def fit(self ,X, y):
        n_samples , n_featuers = X.shape
        self.weight = np.zeros(n_featuers)
        self.bias = 0 

        for _ in range(self.n_iters):
            liner_prediction = np.dot(X,self.weight)+self.bias
            predictions = sigmoid(liner_prediction)

            dw = (1/n_samples) * np.dot(X.T , (predictions-y))
            db = (1/n_samples) * np.sum(predictions-y)
            self.weight = self.weight - self.lr * dw  
            self.bias = self.bias = self.lr * db 


    
    def predict(self , X ):
        liner_prediction = np.dot(X,self.weight)+self.bias
        y_pred = sigmoid(liner_prediction)
        class_predictions = [0 if y<=.5 else 1 for  y in y_pred]
        return class_predictions

