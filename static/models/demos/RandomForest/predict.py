import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pickle

filename = 'trained_model.mdl'
# load the model from disk
model = pickle.load(open(filename, 'rb'))

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y += (0.5 - rng.rand(*y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=400, test_size=200, random_state=4)


y_multirf =model.predict(X_test)
nr=y_multirf.shape[0]
nc=y_multirf.shape[1]


result="Prediction result:\n\n"
for r in range(nr):
    slist=[str(round(y_multirf[r][0],3)),str(round(y_multirf[r][1],3))]
    ss=','.join(slist)
    result+=(ss+'\n')
    
fid=open('result.dat','w')
fid.write(result)
fid.close()

