import numpy as np 


class Perceptron:
    def __init__(self, learning_rate = 0.01 , n_iters = 1000):
        self.n_iters = n_iters
        self.lr = learning_rate
        self.activation_function = self.unit_activashin_f
        self.weights = None
        self.bias = None

    def fit (self , X,y):
        n_sampels , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 
        y_ = np.array([1 if i<0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx , x_i in enumerate(X):
                linear_output = np.dot(x_i , self.weights) + self.bias
                y_predict= self.activation_function(linear_output) 

                update = self.lr * (y[idx] - y_predict)
                self.weights += update * x_i
                self.bias += update 
        
    

    def predict(self,X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predict = self.activation_function(linear_output)
        return y_predict
        

    
        
    

    def unit_activashin_f(self , x):
        return np.where(x>=0 , 1 , 0)
    

