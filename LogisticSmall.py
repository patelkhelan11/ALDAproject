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
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from sklearn import preprocessing

#Reading CSV file as Panda frame
DefaultData=pd.read_csv("bank-additional-full-csv.csv")

#Encoding and deleting the attributes which are not required. Encoding was done just to check the model.
le = preprocessing.LabelEncoder()

le.fit(DefaultData['job'])
DefaultData['job']=le.transform(DefaultData['job'])
del DefaultData['job']

del DefaultData['age']

le.fit(DefaultData['marital'])
DefaultData['marital']=le.transform(DefaultData['marital'])
del DefaultData['marital']

le.fit(DefaultData['default'])
DefaultData['default']=le.transform(DefaultData['default'])
del DefaultData['default']

le.fit(DefaultData['housing'])
DefaultData['housing']=le.transform(DefaultData['housing'])
del DefaultData['housing']

le.fit(DefaultData['loan'])
DefaultData['loan']=le.transform(DefaultData['loan'])
del DefaultData['loan']

le.fit(DefaultData['education'])
DefaultData['education']=le.transform(DefaultData['education'])
del DefaultData['education']

le.fit(DefaultData['contact'])
DefaultData['contact']=le.transform(DefaultData['contact'])
del DefaultData['contact']

le.fit(DefaultData['month'])
DefaultData['month']=le.transform(DefaultData['month'])
#del DefaultData['month']

le.fit(DefaultData['day_of_week'])
DefaultData['day_of_week']=le.transform(DefaultData['day_of_week'])
del DefaultData['day_of_week']

le.fit(DefaultData['poutcome'])
DefaultData['poutcome']=le.transform(DefaultData['poutcome'])
#del DefaultData['poutcome']

del DefaultData['campaign']
del DefaultData['previous']
del DefaultData['cons.price.idx']
del DefaultData['cons.conf.idx']

#y is our output label
y=DefaultData['y']
del DefaultData['y']

#creating training and testing set (80% and 20%), where y is the class variable
#splitting the dataset into training and testing set
DefaultTrain, DefaultValidaiton, y_train, y_test = train_test_split(DefaultData,y,test_size=0.2,random_state=42)
model=LogisticRegression(penalty="l1",C=1)
model.fit(DefaultTrain,y_train)
y_predi=pd.DataFrame(model.predict(DefaultValidaiton))
print('Accuracy is ',accuracy_score(y_test,y_predi))
y_predi=y_predi.replace(['yes', 'no'], [1, 0])
y_test=pd.DataFrame(y_test)
y_test=y_test.replace(['yes', 'no'], [1, 0])

#plotting ROC curve and AUC score and confusion matrix
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predi,pos_label=1)
roc_auc = auc(fpr, tpr)

print(confusion_matrix(y_test, y_predi))

plt.plot(fpr,tpr, 'b')
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Receiver operating characteristic (ROC)')

plt.show()

