print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#   Modified: Jianbo Huang
# License: BSD 3 clause (C) INRIA


# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import pickle
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
#T = np.linspace(0, 5, 500)
tlist=list(T)
fname = 'trained_model.mdl'
with open(fname, 'rb') as f:
   knn=pickle.load(f)
y = knn.predict(T)
result="Prediction result:\n\nx  :   y\n"
for i in range(len(tlist)) :
    result+=str(tlist[i][0])+' '+str(y[i])+"\n"
    
fid=open('result.dat','w')
fid.write(result)
fid.close()

    
   
