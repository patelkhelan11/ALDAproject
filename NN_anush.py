import logging
from sklearn.metrics import confusion_matrix
from sknn.mlp import Classifier, Layer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score
import numpy
logging.basicConfig()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
mydata = pd.read_csv("/Users/anushbonam/Desktop/telemarketing/bank-additional/bank-additional-full.csv", sep=";")

del mydata['job']
del mydata['marital']
del mydata['default']
del mydata['housing']
del mydata['loan']
del mydata['education']
del mydata['contact']
del mydata['month']
del mydata['day_of_week']
del mydata['poutcome']

# data conversion and normalization
mydata = mydata.replace(['yes', 'no'], [1, 0])


# taking the class variable in another column
y = mydata['y']
del mydata['y']
mynewdata = preprocessing.normalize(mydata)

# creating a model and splitting data set into training and testing
DefaultTrain, DefaultValidaiton, y_train, y_test = train_test_split(mynewdata, y, test_size=0.2, random_state=42)

nn = Classifier(layers=[
        Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.003,
    n_iter=25)
nn.fit(DefaultTrain, y_train)
y_valid = nn.predict(DefaultValidaiton)
print('Accuracy: ',nn.score(DefaultValidaiton, y_test))
print confusion_matrix(y_test,y_valid)
fpr, tpr, thresholds =metrics.roc_curve(y_test, y_valid,pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()