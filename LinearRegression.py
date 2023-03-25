import numpy as np 



class LinearRegression():
    def __init__(self , learinig_rate = 0.001 , n_iters = 1000 ):

        self.lr = learinig_rate 
        self.n_iters = n_iters
        self.weights = None
        self.bais = None
    

    def fit(self ,X ,y ):
        
        n_sampels , n_featuers  = X.shape
        self.weights = np.zeros(n_featuers)
        self.bais = 0  

        for _ in  range(self.n_iters):
            y_pred = np.dot(X,self.weights) + self.bais
            dw = (1/n_sampels) * np.dot(X.T,(y_pred - y ))
            db = (1/n_sampels) * np.sum(y_pred - y )

            self.weights = self.weights - self.lr * dw 
            self.bais = self.bais - self.lr * db 


    def predict(self , X):
        y_pred = np.dot(X,self.weights) + self.bais
        return y_pred
        