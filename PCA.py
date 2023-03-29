import numpy as np

class PCA:
    def __init__(self , n_components ):
        self.n_components = n_components
        self.component = None
        self.mean = None
    

    def fit(self , X):
        
        self.mean = np.mean(X , axis=0 )
        X = X - self.mean
        cov = np.cov(X.T)
        enginevalues , enginevictors = np.linalg.eig(cov)
        enginevictors = enginevictors.T
        idxs = np.argsort(enginevalues)[::-1]
        enginevalues = enginevalues[idxs]
        enginevictors = enginevictors[idxs]


        self.component = enginevictors[:self.n_components]
         



    def transform(self, X):
        X = X - self.mean
        component = self.component.T
        return np.dot(X ,component) 