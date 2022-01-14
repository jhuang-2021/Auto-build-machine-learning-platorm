import os
from sklearn import model_selection
#from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import pickle
import numpy as np

fid=open('input.csv','r')
clist=fid.read().split('\n')
fid.close()
hdr=clist[0].split(',')
ncol=len(hdr)
test_size = 0.18
n_neighbors = 7

trainData=[]
for r in range(1,len(clist)):
    ss=clist[r]
    if len(ss)<10: continue #skip empty lines
    slist=ss.split(',')
    dlist=[]
    for c in range(len(slist)):
        sd=slist[c]
        try: d=float(sd)
        except: 
            d=0
            print('not a numerical data',r,c)
        dlist.append(d)
    trainData.append(dlist)

XY=np.array(trainData)

X=XY[:,0:ncol-1]

Y=XY[:,ncol-1]
#normalize for better accuracy
xmax=[]
for c in range(ncol-1):
    amax=np.max(X[:,c])
    X[:,c]/=amax

seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
#print(len(Y_test))
weights='distance'

knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
model=knn.fit(X_train, Y_train)
fname = 'trained_model.mdl'
pickle.dump(model, open(fname, 'wb'))

y_pred =model.predict(X_test)
ncorrect=0
ntest=len(y_pred)
for i in range(ntest):
    if y_pred[i]==Y_test[i]: ncorrect+=1
    
ratio=float(ncorrect)/float(ntest)    
result="Test report\n"
result+= "Correct_predict    Test_number     Correct_rate\n"   
result+= (str(ncorrect)+"   "+str(ntest)+"   "+str(round(ratio,3))+'\n')
print(ncorrect,len(y_pred),ratio)

tname='test_report.txt'
ff=open(tname,'w')
ff.write(result)
ff.close()
