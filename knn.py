import numpy as np
from collections import Counter

def decipher(Values):
    fruit_dict = {
    1 : "apple",
    2 : "Mandarin",
    3 : "Orange",
    4 : "Lemon"
    }
    to_return = [fruit_dict[x] for x in Values]
    return to_return

def Distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def accuracy(y_actual,y_pred):
    accuracy = np.sum(y_actual == y_pred) / len(y_actual)
    return accuracy

class KNN:
    def __init__(self,k=3):
        self.k = 3

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X):
        predicted = [self.indiv_predict(item) for item in X]
        return np.array(predicted)
    
    def indiv_predict(self,x):
        distances = [Distance(x, entry) for entry in self.X]
        nearest_neighbour_indices = np.argsort(distances)[0:self.k]
        nearest_neighbour_labels = self.y[nearest_neighbour_indices]
        most_common = Counter (nearest_neighbour_labels).most_common(1)
        return most_common[0][0]