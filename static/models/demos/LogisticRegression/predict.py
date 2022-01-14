import pickle
import pandas
from sklearn import model_selection
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
filename = 'trained_model.mdl'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
res = loaded_model.score(X_test, Y_test)

result="Prediction result:\n\n"
result+="Score: "+str(res)+"\n"
fid=open('result.dat','w')
fid.write(result)
fid.close()
