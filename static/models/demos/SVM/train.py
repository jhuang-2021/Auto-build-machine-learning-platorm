import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pickle

iris = datasets.load_iris()
X = iris.data[:, :2] 

y = iris.target

C = 1.0 
svc = svm.SVC(kernel='linear', C=1,gamma=0.7).fit(X, y)
filename = 'trained_model.mdl'
pickle.dump(svc, open(filename, 'wb'))
