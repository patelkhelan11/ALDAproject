import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve,auc
import pandas as pd
import csv
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
xl = pd.read_excel('my_data.xls')
columns = xl.columns.values.tolist()
my_data = xl.as_matrix(columns)
row_count = my_data[:,0].size
train_index = int(row_count * 0.7)
X = my_data[:,:7]
y = my_data[:,7]
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.20,random_state=0)

C_range = np.arange(1, 4)

param_grid = dict(gamma=[0.0005], C=[1])
svr = svm.SVC()
model = GridSearchCV(svr, param_grid)

model.fit(train_X, train_Y)
print("The best classifier is: ", model.best_estimator_)
print model.score(train_X, train_Y)
predicted = model.predict(test_X)

print model.score(test_X, test_Y)
print confusion_matrix(test_Y, predicted)
fpr, tpr, thresholds =metrics.roc_curve(test_Y, predicted,pos_label=1)
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr, 'b')
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Receiver operating characteristic (ROC)')

plt.show()

