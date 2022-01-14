import os
import pickle
import numpy as np
import sys

hasLabel=True
nparm=len(sys.argv)

gid=open('input.csv','r')
glist=gid.read().split('\n')
gid.close()    
ghdr=glist[0].split(',')
gncol=len(ghdr)
        
fid=open('predict.csv','r')
clist=fid.read().split('\n')
fid.close()
hdr=clist[0].split(',')
ncol=len(hdr)
hasLabel=True
if ncol<gncol: hasLabel=False

gData=[]
for r in range(1,len(glist)):
    ss=glist[r]
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
    gData.append(dlist)

gXY=np.array(gData)
xgmax=[]
for c in range(gncol-1):
    amax=np.max(gXY[:,c])
    xgmax.append(amax)
    
allData=[]
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
    allData.append(dlist)

XY=np.array(allData)
nend=ncol-1
if not hasLabel: nend=ncol

X=XY[:,0:nend]
print('ncol,xshape',ncol,gncol,X.shape[1])
print(xgmax)


#Y=XY[:,ncol-1]
#normalize for better accuracy
xmax=[]
for c in range(nend):
    amax=np.max(X[:,c])
    amax=max(amax,xgmax[c])
    X[:,c]/=amax

filename = 'trained_model.mdl'
model = pickle.load(open(filename, 'rb'))

y_pred=list(model.predict(X))
nr=len(y_pred)

result='Prediction result\n'
hdr=clist[0]

if not hasLabel:
    hdr+=',DEATH_EVENT'
result+=(hdr+'\n')
for r in range(nr):
    ss=clist[r+1]
    slist=ss.split(',')
    klist=slist[0:nend]  
    ypt=int(y_pred[r]+0.01)  
    klist.append(str(ypt))
    s1=','.join(klist)
    result+=(s1+'\n')
tname='predict.txt'
ff=open(tname,'w')
ff.write(result)
ff.close()





