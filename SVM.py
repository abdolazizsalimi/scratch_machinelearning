import numpy as np 
import sys
sys.dont_write_bytecode = True
class SVM:
    def __init__(self, learnig_rate =0.001 , lambda_param=0.02 ,n_iters = 2000):
        self.lr = learnig_rate 
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
        


    def fit(self ,X ,y ):
        _y = np.where(y<=0 ,-1 , 1 )
        x_sampel , n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0 

        for _ in range(self.n_iters):
            for idx , x_i in enumerate(X):
                condition = _y[idx] * (np.dot(x_i , self.weight)-self.bias) >= 1
                if condition : 
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight)
                else :
                    self.weight -= self.lr * (2*self.lambda_param*self.weight)- np.dot(x_i ,_y[idx])
                    self.bias -= self.lr * _y[idx]     
                    


        
         
    
    def predict(self,X):
        linear_output = np.dot(X , self.weight) - self.bias
        return np.sign(linear_output) 
