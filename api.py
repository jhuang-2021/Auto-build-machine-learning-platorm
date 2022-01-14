#!/usr/bin/python
# coding=utf8


from flask import Flask, make_response,url_for, render_template, request, redirect, session,Response
import os
#import numpy as np
import pathlib
from flask_sqlalchemy import SQLAlchemy #database interface
import pickle
from os import listdir
from os.path import isfile, join
import datetime
from os import path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import requests
import re
import shutil
from subprocess import call
#import time

def read_csv(fname='trainData.csv'):
    fid=open(fname,'r')
    if not fid:
        print('error read csv file: specified file cannot open')
        return None
    datalist=fid.read().split('\n')
    data=[]
    for ss in datalist:
        if len(ss)<10: continue
        slist=ss.split(',')
        data.append(slist)
    fid.close()
    return data
    
def save_csv(datalist,fname='trainDataNew.csv'):
    fid=open(fname,'w')
    if not fid:
        print('error save csv file: specified file cannot open')
        return None
    result=''
    for slist in datalist:
        rlist=[str(ss) for ss in slist]
        sr=','.join(rlist)
        result+=(sr+'\n')
    fid.write(result)
    fid.close()
    return True    




def getuser():
    uname=session.get('user')
    if not uname: return None
    user=User.query.filter_by(username=uname).first()
    return user

def getuid():
    user=getuser()
    if not user: return -1
    return user.id

def getdate():
  currentDT = datetime.datetime.now()
  t=str(currentDT)
  tlist=t.split(" ")
  return tlist[0]

def lastid(obj):
      gid=obj.query.all()
      if not gid: return 0
      id=-1
      if len(gid)==0:return 0
      if gid is not None: id=gid[-1].id
      return id

def exists(path):
    if os.path.isfile (path) : return True
    else: return False
    
application = Flask(__name__)
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mlapp.db'
db = SQLAlchemy(application)

class User(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   status=db.Column(db.Integer)
   
   username = db.Column(db.String(), unique=True,index=True)
   email=db.Column(db.String())
   phone=db.Column(db.String())
   mobile=db.Column(db.String())
   password = db.Column(db.String())
   dept=db.Column(db.String(),index=True)
   join=db.Column(db.String())
   org=db.Column(db.String(),index=True)
   
   def __init__(self, username, password,dept="industry",phone="none",mobile="none",org="none",email="none"):
      self.username = username
      self.password = password
      self.email=email
      self.phone=phone
      self.mobile=mobile
      self.dept=dept
      self.status=1
      self.join=getdate()
      self.org=org
     
   def mydir(self):
       cwd=os.path.getcwd()
       dname=os.path.join(cwd,'static','users',str(self.id))
       return dname
  

class MLModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name=db.Column(db.String())
    desc=db.Column(db.String())
    mdltype=db.Column(db.String())
    parms=db.Column(db.String())
    creator=db.Column(db.String(),index=True)
    create_date=db.Column(db.String(30))
    dname=db.Column(db.String())
    def __init__(self, name,mdltype,desc='',parms=''):
        self.name=name
        self.create_date=getdate()
        self.creator=session.get('user')
        self.desc=desc
        self.mdltype=mdltype
        self.parms=parms
        user=getuser()
        pid=lastid(MLModel)+1
        self.dname=os.path.join('static','users',str(user.id),'mlapp',str(pid))
        pathlib.Path(self.dname).mkdir(parents=True, exist_ok=True)
    
    def copyModel(self):
        mdlMap={'K-Nearest Neighbour':'KNN','Decision Tree':'DecisionTree',
            'Support Vector Machine':'SVM','Multilayer Perceptron':'MLP',
            'Random Forest':'RandomForest'}
        mdl=mdlMap[self.mdltype]
        mdlfile=os.path.join('static','models','medical',mdl,'train.py')
        shutil.copy2(mdlfile,self.dname)
    def copyPredict(self):
        mdlfile=os.path.join('static','models','medical','predict.py')
        shutil.copy2(mdlfile,self.dname)
    
    def setParms(self,parms):
        self.parms=parms
    def copyInput(self):
        user=getuser()
        dname= self.dname
        csvfile= os.path.join('static','models','medical','input.csv')  
        shutil.copy2(csvfile,dname)
    def copyPredictData(self):
        user=getuser()
        csvfile= os.path.join('static','models','medical','predict.csv')  
        shutil.copy2(csvfile,self.dname)    
    def dirName(self):
        return self.dname
        
welcome='Welcome to use MLApp: cloud-based machine-learning applications'
info={'topic':'none','msg':welcome,'user':'none'}
Info={}

@application.route('/', methods=['GET', 'POST'])
def home():
    id=getuid()
    if id not in Info.keys():
        Info[id]={}
    info=Info[id]
    info['topic']='none'
    info['msg']=welcome
    if not session.get('logged_in'):
        return render_template('index.html',info=info)
    else:
        if request.method == 'POST':
            username = getname(request.form['username'])
            return render_template('index.html')
    return render_template('index.html',info=info)




@application.route('/regist' , methods=['GET', 'POST'])
def regist():
    id=getuid()
    if id not in Info.keys():Info[id]={}
    info=Info[id]
    if id>=0:
        info['msg']="You already registered. Please choose another service"
        info['topic']='none'
        return render_template('index.html',info=info)
    info['topic']='regist'
    info['msg']=welcome
    if request.method == 'GET':
        return render_template('index.html',info=info)
    button=request.form.get('button')
   
    email=request.form.get('email')
    wno=email
   
    org="public"
    dept="individual"
        
    data=User.query.filter_by(username=wno).first()
    if data is not None:
        f="Error: user email already in use. Please enter another:"
        info['msg']=f
        return render_template('index.html',info=info)
   
    passwd=request.form['password']
    passwd2=request.form['password2']
        
    if passwd !=passwd2:
        f="Error: email and repeated email must be the same"
        info['msg']=f
        return render_template('index.html',info=info)
    if (len(passwd)<6):
        f="Error: password length must be >= 6"
        info['msg']=f
        return render_template('index.html',info=info)
    
    new_user = User(username=wno, password=passwd,dept=dept,org=org,email=email)
    db.session.add(new_user)
    db.session.commit()
    id=lastid(User)
    
    dname=os.path.join('static','users',str(id)) 
   
    pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
    
    session['logged_in'] = True
    session['user'] = wno
    session['isadmin'] = False
    f="Welcome "+wno+" to use MLApp"
    info['msg']=f
    info['topic']='about'
    return render_template('index.html',info=info)

@application.route('/logout' , methods=['GET', 'POST'])
def logout():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]
    if 'logged_in' not in session.keys():
        info['msg']='Logout Error：you are not logged in'
        info['topic']='login'
        return render_template('index.html',info=info)      
    if not session['logged_in'] :
        info['msg']='Logout Error：you are not logged in'
        info['topic']='login'
        return render_template('index.html',info=info)   
    f="You are logged out"
    session['logged_in'] = False
    session['user'] =None
    session['isadmin'] = False
    info['msg']=f  
    info['topic']='none'
    return render_template('index.html',info=info)         
 
@application.route('/predict' , methods=['GET', 'POST'])
def predict():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='predict'  
    info['msg']='Prdiction use a trained machine-learning model'  
    if not user:
        msg='Error predict an outcome with machine-learning, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info) 
    models=MLModel.query.filter_by(creator=user.username).all()
    if len(models)==0:
        msg='Error doing prediction : you do not have any model in database. Please set the model first'
        info['msg']=msg
        return render_template('index.html',info=info)
    mdls=[['id','Model Name','Creation date','type','creator','description']]
    for mdl in models:
        mlist=[mdl.id,mdl.name,mdl.create_date,mdl.mdltype,mdl.creator,mdl.desc]
        mdls.append(mlist)
        
    info['model_list']=mdls 
    if request.method == 'GET':
        msg='Use a trained machine-learning model to do prediction'
        info['msg']=msg
        return render_template('index.html',info=info)
    
    button=request.form.get('button')
    mid=request.form.get('mdlID')
    info['mdlID']=mid
    
    if button=='Cancel':
        msg='Action for prediction cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info) 
    if not mid:
        msg='Error doing prediction: model id not specified'
        info['msg']=msg
        return render_template('index.html',info=info) 
    try: id=int(mid)
    except Exception as e:
        msg='Error doing prediction, error message is:'+str(e)
        info['msg']=msg
        return render_template('index.html',info=info)    
    model= MLModel.query.filter_by(id=id,creator=user.username).first()
    if not model:
        msg='Error doing prediction: specified model id not exists or not owned by you'
        info['msg']=msg
        return render_template('index.html',info=info)      
    # copy model file
    model.copyPredict()
    dname=model.dirName()
    
    cwd=os.getcwd()
    # change working directory
    os.chdir(dname)
    cmd='python predict.py'
    os.system(cmd)
    #lcmd=["python",'predict.py']
    #call(lcmd)
    if not exists('predict.txt'):
        msg='Error doing prediction: no predict.txt file found in the working directory'
        info['msg']=msg
        return render_template('index.html',info=info)   
    else:
        fd=open('predict.txt','r')
        result=fd.read()
        fd.close()
        info['result']=result 
    os.chdir(cwd)
    msg='Machine-learning prediction job completed'
    info['msg']=msg
    return render_template('index.html',info=info)   
     
@application.route('/train_model' , methods=['GET', 'POST'])
def train_model():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='train_model'  
    info['msg']='Train a machine-learning model'  
    if not user:
        msg='Error train a machine-learning model, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info) 
    models=MLModel.query.filter_by(creator=user.username).all()
    if len(models)==0:
        msg='Error train a model: you do not have any model in database. Please set the model first'
        info['msg']=msg
        return render_template('index.html',info=info)
    mdls=[['id','Model Name','Creation date','type','creator','description']]
    for mdl in models:
        mlist=[mdl.id,mdl.name,mdl.create_date,mdl.mdltype,mdl.creator,mdl.desc]
        mdls.append(mlist)
        
    info['model_list']=mdls 
    if request.method == 'GET':
        msg='Train a machine-learning model'
        info['msg']=msg
        return render_template('index.html',info=info)
    
    button=request.form.get('button')
    mid=request.form.get('mdlID')
    info['mdlID']=mid
    
    if button=='Cancel':
        msg='Train a machine-learning model action cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info) 
    if not mid:
        msg='Error training a machine-learning model: model id not specified'
        info['msg']=msg
        return render_template('index.html',info=info) 
    try: id=int(mid)
    except Exception as e:
        msg='Error training machine-learning model for case, error message is:'+str(e)
        info['msg']=msg
        return render_template('index.html',info=info)    
    model= MLModel.query.filter_by(id=id,creator=user.username).first()
    if not model:
        msg='Error training machine-learning model: specified model id not exists or not owned by you'
        info['msg']=msg
        return render_template('index.html',info=info)      
    # copy model file
    model.copyModel()
    dname=model.dirName()
    
    cwd=os.getcwd()
    # change working directory
    os.chdir(dname)
    cmd='python train.py'
    #os.system(cmd)
    lcmd=["python",'train.py']
    call(lcmd)
    os.chdir(cwd)
    msg='Train model job is running in the background, it may take some time depending on models'
    info['msg']=msg
    return render_template('index.html',info=info)   
    
@application.route('/getModel' , methods=['GET', 'POST'])
def getModel():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='getModel'  
    info['msg']='Get trained model file'  
    if not user:
        msg='Error get trained model file, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info) 
    models=MLModel.query.filter_by(creator=user.username).all()
    if len(models)==0:
        msg='Error get trained model file: you do not have any model in database. Please set the model first'
        info['msg']=msg
        return render_template('index.html',info=info)
    mdls=[['id','Model Name','Creation date','type','creator','description']]
    for mdl in models:
        mlist=[mdl.id,mdl.name,mdl.create_date,mdl.mdltype,mdl.creator,mdl.desc]
        mdls.append(mlist)
        
    info['model_list']=mdls 
    if request.method == 'GET':
        msg='Download trained model file'
        info['msg']=msg
        return render_template('index.html',info=info)
    button=request.form.get('button')
    mid=request.form.get('mdlID')
    info['mdlID']=mid
    
    if button=='Cancel':
        msg='Download trained model action cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info) 
    if not mid:
        msg='Error download trained model: model id not specified'
        info['msg']=msg
        return render_template('index.html',info=info) 
    try: id=int(mid)
    except Exception as e:
        msg='Error download trained model for case, error message is:'+str(e)
        info['msg']=msg
        return render_template('index.html',info=info)    
    model= MLModel.query.filter_by(id=id,creator=user.username).first()
    if not model:
        msg='Error download trained model: specified model id not exists or not owned by you'
        info['msg']=msg
        return render_template('index.html',info=info)      
    dname=model.dirName()
    fname=os.path.join(dname,'trained_model.mdl')
    if not exists(fname):
        msg='Error download trained model: trained model for this case does not exist'
        info['msg']=msg
        return render_template('index.html',info=info)    
    
    tgtfile='trained_model.mdl'
    try:
        fs=open(fname,'rb')
        contents=fs.read()
        response = make_response(contents)
        cd = 'attachment; filename='+tgtfile
        response.headers['Content-Disposition'] = cd 
        response.mimetype='text/csv'
        return response
    except Exception as e:
        info['msg']='Error download trained model, error message :'+str(e)
        return render_template('index.html',info=info) 

@application.route('/testReport' , methods=['GET', 'POST'])
def testReport():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='testReport'  
    info['msg']='Get test report for model training'  
    if not user:
        msg='Error get test report for model training, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info) 
    
    models=MLModel.query.filter_by(creator=user.username).all()
    if len(models)==0:
        msg='Error get test report for model training: you do not have any model in database. Please set the model first'
        info['msg']=msg
        return render_template('index.html',info=info)
    mdls=[['id','Model Name','Creation date','type','creator','description']]
    for mdl in models:
        mlist=[mdl.id,mdl.name,mdl.create_date,mdl.mdltype,mdl.creator,mdl.desc]
        mdls.append(mlist)
        
    info['model_list']=mdls
    
    if request.method == 'GET':
        msg='get test report for model training'
        info['msg']=msg
        return render_template('index.html',info=info)
    button=request.form.get('button')
    mid=request.form.get('mdlID')
    info['mdlID']=mid
    
    if button=='Cancel':
        msg='Get test report for model training action cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info) 
    
    if not mid:
        msg='Error get test report for model training: please specify model id'
        info['msg']=msg
        return render_template('index.html',info=info) 
        
        
    try: id=int(mid)
    except Exception as e:
        msg='Error get test report for model training: '+str(e)
        info['msg']=msg
        return render_template('index.html',info=info)    
    model= MLModel.query.filter_by(id=id,creator=user.username).first()
    if not model:
        msg='Error get test report for model training: specified model id not exists or not owned by you'
        info['msg']=msg
        return render_template('index.html',info=info)           
    
    if button=='Get report':
        dname= model.dirName()
        fname=os.path.join(dname,'test_report.txt')
        fd=open(fname,'r')
        rep=fd.read()
        fd.close()
        print('rep',rep)
        msg='Test report is shown'
        info['msg']=msg  
        info['report']=rep  
        return render_template('index.html',info=info)  
    return render_template('index.html',info=info) 

@application.route('/setPredict' , methods=['GET', 'POST'])
def setPredict():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='setPredict'  
    info['msg']='Set prediction input data'  
    if not user:
        msg='Error set prediction data, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info) 
    
    models=MLModel.query.filter_by(creator=user.username).all()
    if len(models)==0:
        msg='Error set prediction data: you do not have any model in database. Please set the model first'
        info['msg']=msg
        return render_template('index.html',info=info)
    mdls=[['id','Model Name','Creation date','type','creator','description']]
    for mdl in models:
        mlist=[mdl.id,mdl.name,mdl.create_date,mdl.mdltype,mdl.creator,mdl.desc]
        mdls.append(mlist)
        
    info['model_list']=mdls
    
    if request.method == 'GET':
        msg='Set model prediction data for '+user.username
        info['msg']=msg
        return render_template('index.html',info=info)
    button=request.form.get('button')
    mid=request.form.get('mdlID')
    info['mdlID']=mid
    
    if button=='Cancel':
        msg='Set prediction data action cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info) 
    
    if not mid:
        msg='Error set prediction data: please specify model id'
        info['msg']=msg
        return render_template('index.html',info=info) 
        
        
    try: id=int(mid)
    except Exception as e:
        msg='Error set prediction data: '+str(e)
        info['msg']=msg
        return render_template('index.html',info=info)    
    model= MLModel.query.filter_by(id=id,creator=user.username).first()
    if not model:
        msg='Error set prediction data: specified model id not exists or not owned by you'
        info['msg']=msg
        return render_template('index.html',info=info)           
    
    if button=='Get data':
        predict=request.form.get('predict')
        predict=predict.strip()
        info['predict']=predict
        dname= model.dirName()
        fname=os.path.join(dname,'predict.csv')
        fd=open(fname,'w')
        fd.write(predict)
        fd.close()
        
        msg='Model prediction data saved'
        info['msg']=msg    
        return render_template('index.html',info=info)  
    
    return render_template('index.html',info=info) 

@application.route('/load_data' , methods=['GET', 'POST'])
def load_data():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='load_data'  
    info['msg']='Loading training data files. File must be in CSV format'  
    if not user:
        msg='Error loading training data, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info) 
    models=MLModel.query.filter_by(creator=user.username).all()
    if len(models)==0:
        msg='Error load ML training data: you do not have any model in database. Please set the model first'
        info['msg']=msg
        return render_template('index.html',info=info)
    mdls=[['id','Model Name','Creation date','type','creator','description']]
    for mdl in models:
        mlist=[mdl.id,mdl.name,mdl.create_date,mdl.mdltype,mdl.creator,mdl.desc]
        mdls.append(mlist)
        
    info['model_list']=mdls
    
    if request.method == 'GET':
        msg='Load ML training data for '+user.username
        info['msg']=msg
        return render_template('index.html',info=info)
    button=request.form.get('button')
    mid=request.form.get('mdlID')
    info['mdlID']=mid
    
    if button=='Cancel':
        msg='Load ML training data action cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info) 
    
    url=request.form.get('url')   
    info['url']=url
    
    if not mid:
        msg='Error load ML training data: please specify model id'
        info['msg']=msg
        return render_template('index.html',info=info) 
        
        
    try: id=int(mid)
    except Exception as e:
        msg='Error load ML training data: '+str(e)
        info['msg']=msg
        return render_template('index.html',info=info)    
    model= MLModel.query.filter_by(id=id,creator=user.username).first()
    if not model:
        msg='Error load ML training data: specified model id not exists or not owned by you'
        info['msg']=msg
        return render_template('index.html',info=info)           
    
    if button=='Load file':
        ext=".csv"
        f = request.files.get("samplefile")   
        if not f:
            msg='Error load ML training data: file not specified'
            info['msg']=msg
            return render_template('index.html',info=info) 
        dname= model.dirName()
        ofile=f.filename 
        exts=os.path.splitext(ofile)
        if exts[1]!=ext:
            msg='Error load input file: must be with extension .csv'
            info['msg']=msg    
            return render_template('index.html',info=info)   
        fname=os.path.join(dname,'input.csv')
        
        f.save(fname)  
        msg='Specified file saved'
        info['msg']=msg    
        return render_template('index.html',info=info)  
    if button=='Get from url':
        if not url:
            msg='Error get data from URL: address not specified'
            info['msg']=msg
            return render_template('index.html',info=info) 
        # validate URL
        
        regex = re.compile(
        r'^(?:http|ftp)s?://' 
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' 
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' 
        r'(?::\d+)?' 
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if re.match(regex, url) is None:
            msg='Error get training data from URL: invalid URL'
            info['msg']=msg
            return render_template('index.html',info=info) 
        r = requests.get(url, allow_redirects=True)
        scnt=r.content
        dname=model.dirName()
        dfile='input.csv'
        fname=os.path.join(dname,dfile)
        fid=open(fname, 'w')
        fid.write(scnt)
        fid.close()
        msg='Training data obtained from URL'
        info['msg']=msg
    return render_template('index.html',info=info) 
        
@application.route('/set_model' , methods=['GET', 'POST'])
def set_model():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['topic']='set_model'
    info['modelType']='K-Nearest Neighbour'
    if not user:
        msg='Error set machine-learning model, you need to login to your account first'
        info['msg']=msg
        return render_template('index.html',info=info)  
    if request.method == 'GET':
        msg='Setup machine-learning model for '+user.username
        info['msg']=msg
        return render_template('index.html',info=info)
    button=request.form.get('button')
    if button=='Cancel':
        msg='Set machine-learning model action cancelled'
        info['msg']=msg
        info['topic']='none'
        return render_template('index.html',info=info)  
    
    model_name=request.form['model_name']
    info['model_name']=model_name
    modelType=request.form['modelType']
    info['modelType']=modelType
    model_desc=request.form['model_desc']
    info['model_desc']=model_desc
    
    #(self, name,mdltype,desc='')
    mlapp1=MLModel.query.filter_by(name=model_name,creator=user.username).first()
    if mlapp1:
        msg='Error set ML model: specified name already in use'
        info['msg']=msg
        return render_template('index.html',info=info)
    mlapp=MLModel(model_name,modelType,model_desc)
    if button=='Apply':
        mlapp.copyModel()
        mlapp.copyPredict()
        mlapp.copyInput()
        mlapp.copyPredictData()
        
        db.session.add(mlapp)
        db.session.commit()
        msg='Specified machine-learning model created'
        info['msg']=msg
        return render_template('index.html',info=info)
    return render_template('index.html',info=info)

@application.route('/user_login' , methods=['GET', 'POST']) 
def user_login():
    user=getuser()
    if not user: id=0        
    else: id=user.id
    if id not in Info.keys():Info[id]={}
    info=Info[id]    
    info['msg']=welcome
    info['topic']='user_login'
    
    if user:
        info['msg']='You already logged in'
        info['topic']='none'
        return render_template('index.html',info=info)
    
    if request.method == 'GET':
        return render_template('index.html',info=info)
    
    name = request.form['username']
    passw = request.form['password']
    user=User.query.filter_by(username=name).first()
    
    try:
        data = User.query.filter_by(username=name, password=passw).first()
        fname = User.query.filter_by(username=name).first()
        if fname is None:
            f="Error: user name not exists. Please register first"
            info['msg']=f
            return render_template('index.html',info=info)
        if data is not None:
            id=data.id
            sid=str(id)
            session['logged_in'] = True
            session['isadmin'] = False
            session['user'] = name
            session['uid']=sid
            session['group'] = False
            f="Welcome "+name+" to use WebDAT"
            info['msg']=f
            info['topic']='none'
            return render_template('index.html',info=info)
        else:
            f="Error login: incorrect password"
            info['msg']=f
            return render_template('index.html',info=info)
    except:
        f="Error: incorrect password"
        info['msg']=f
        return render_template('index.html',info=info)
    info['topic']='none'   
    return render_template('index.html',info=info)


application.debug = False
db.create_all()
application.secret_key = "mlapp-2021-05-15@*+U"
if __name__ == '__main__':
   
   #set host = server ip to run on server
   application.run(host='0.0.0.0')
