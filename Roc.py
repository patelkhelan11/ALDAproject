import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
from sklearn import metrics


#Reading CSV file in Panda frame
DefaultData=pd.read_csv("bank-additional-full-csv.csv")

#encode data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


#print(DefaultData.colum
#Choosing only balance and income as the features to train the model

#del DefaultData['job']
#del DefaultData['marital']
#del DefaultData['default']
#del DefaultData['housing']
#del DefaultData['loan']
#del DefaultData['education']
#del DefaultData['contact']
#del DefaultData['month']
#del DefaultData['day_of_week']
#del DefaultData['poutcome']


y=DefaultData['y']
del DefaultData['y']
lb = preprocessing.LabelBinarizer()
lb.fit_transform(['yes', 'no', 'no', 'yes'])
n_classes = 2
print(y)
#print(DefaultData)
#creating training and testing set, where y is the class variable(Default0
#DefaultTrain, DefaultValidaiton, y_train, y_test = train_test_split(DefaultDataNew,y,test_size=0.25,random_state=42)

#creating a model

DefaultTrain, DefaultValidaiton, y_train, y_test = train_test_split(DefaultData,y,test_size=0.3,random_state=42)
model=LogisticRegression(penalty="l1",C=1)

model.fit(DefaultTrain,y_train)
y_predi=model.predict(DefaultValidaiton)
print(accuracy_score(y_test,y_predi))
#y_test=pd.DataFrame(y_test)


#fpr, tpr, thresholds =metrics.roc_curve(y_test, y_predi,pos_label=1)


#plt.plot(fpr,tpr)
#plt.show()
#print(model.predict(DefaultValidaiton))
#print(confusion_matrix(y_test, model.predict(DefaultValidaiton)))
#print(model.fit(DefaultData,y))
#print (1-accuracy_score(y,model.predict(DefaultData)))
#print(accuracy_score(y, model.predict(DefaultData)))
