import numpy as np 


class NaiveBayes:
    def __init__(self):
        pass 

    
    def fit(self , X,y):
        n_samples , n_features = X.shape 
        self._uniqueClass = np.unique(y)
        n_class = len(self._uniqueClass)
        
        
        self._mean = np.zeros(shape=(n_class ,n_features) , dtype =np.float64)
        self._var = np.zeros(shape=(n_class ,n_features) , dtype =np.float64)
        self._priors = np.zeros(shape=(1,n_class) , dtype= np.float64)

        for c in self._uniqueClass:
            X_class = X[c==y]
            self._mean[c,:] = X_class.mean

    def prediction(X):
        pass 