
import numpy as np
from collections import Counter



class KNN:

    def __init__(self , k = 3 ):
        self.k = k 

        


    def fit(self , X,y):
        self.x_train = X 
        self.y_train = y


    def predict(self, X):
        predict_labels = [self._predict(x) for x in X]
        return np.array(predict_labels)
    

    def _predict(self,x):

        # compute distance 
        distances = [cal_distance(x , i ) for i in self.x_train ]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_commen = Counter(k_nearest_labels).most_common(1)
        return most_commen[0][0]
def cal_distance(x1 ,x2):
    res = np.sqrt(np.sum(x1-x2)**2)
    return res