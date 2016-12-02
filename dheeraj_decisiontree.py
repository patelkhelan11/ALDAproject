from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np

# reading the initial dataset

df = pd.read_csv('bank-additional-full.csv',sep=",")

# columns we want in the final Model
columns_you_want = ['emp.var.rate', 'duration', 'month', 'euribor3m', 'nr.employed', 'pdays', 'poutcome']
end_column = ['y']

# numeric conversion
le = preprocessing.LabelEncoder()

le.fit(df['poutcome'])
df['poutcome']=le.transform(df['poutcome'])

le.fit(df['housing'])
df['housing']=le.transform(df['housing'])

le.fit(df['month'])
df['month']=le.transform(df['month'])

le.fit(df['y'])
df['y']=le.transform(df['y'])

# binning of teh features

bins = [0, 2, 4,6] #"""[0-2] are put as 1,[2-4] are put as 2,[4-6] are put as 3"""#
df['euribor3m'] = np.digitize(df['euribor3m'], bins)

bins = [0, 10, 20,30,999] #"""[0-2] are put as 1,[2-4] are put as 2,[4-6] are put as 3"""#
df['pdays'] = np.digitize(df['pdays'], bins)

bins = [-4, -3, -2,-1,0,1,2] #"""[0-2] are put as 1,[2-4] are put as 2,[4-6] are put as 3"""#
df['emp.var.rate'] = np.digitize(df['emp.var.rate'], bins)

bins = [0,100,200,300,400,500,600,1000,1500,2000,2500,3000,3500,4000,4500,5000] #"""[0-2] are put as 1,[2-4] are put as 2,[4-6] are put as 3"""#
df['duration'] = np.digitize(df['duration'], bins)

bins = [4900, 5000, 5100, 5200, 5300, 5400] #"""[0-2] are put as 1,[2-4] are put as 2,[4-6] are put as 3"""#
df['nr.employed'] = np.digitize(df['nr.employed'], bins)

# splitting the data into Training and Testing data
trainingData = df.sample(frac = 0.8, replace = False)
testData = df.drop(trainingData.index)

# saving it into excel

writer = pd.ExcelWriter('trainingdata.xlsx')
trainingData.to_excel(writer,'Sheet1')
writer.save()

writer = pd.ExcelWriter('testingdata.xlsx')
testData.to_excel(writer, 'Sheet1')
writer.save()

print("excel saved")

# read from the saved files
xlsheet = pd.ExcelFile('trainingdata.xlsx')
x2sheet = pd.ExcelFile('testingdata.xlsx')

df = xlsheet.parse('Sheet1')
df2 = x2sheet.parse('Sheet1')


training_Df =  df[columns_you_want]
training_Predict_Df = df[end_column]

testing_df =  df2[columns_you_want]
testing_predict_alreadygiven = df2[end_column]

# Gini index Decision Tree building
clf = tree.DecisionTreeClassifier(min_samples_split=2000, random_state=1, splitter = "random" )
clf = clf.fit(training_Df, training_Predict_Df)
print clf

# Entropy based Decision Tree building

clf_entropy = tree.DecisionTreeClassifier(min_samples_split=2000, random_state=1, criterion='entropy' , splitter = "random")
clf_entropy = clf_entropy.fit(training_Df, training_Predict_Df)

# used to create the graph of the decision tree
tree.export_graphviz(clf_entropy, out_file='tree.dot', feature_names = columns_you_want)
rclf = RandomForestClassifier(criterion='entropy', min_samples_split=200,  max_features='auto', max_leaf_nodes=None, verbose=0)
rclf =rclf.fit(training_Df, training_Predict_Df.values.ravel())

print "Gini Score: " , clf.score(training_Df, training_Predict_Df)
print "Entropy Score: ", clf_entropy.score(training_Df, training_Predict_Df)
print "RandomForestClassifier: ", rclf.score(training_Df, training_Predict_Df)

# prediciting all the three different trees
testing_df_predicted_values = clf.predict(testing_df)
testing_df_perdicted_values_entropy = clf_entropy.predict(testing_df)
testing_df_predicted_values_randomforest = rclf.predict(testing_df)

#ROC curve for all the three decision treea
fpr, tpr, thresholds = metrics.roc_curve(testing_predict_alreadygiven, testing_df_predicted_values, pos_label=1)
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


fpr, tpr, thresholds = metrics.roc_curve(testing_predict_alreadygiven, testing_df_perdicted_values_entropy, pos_label=1)
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


fpr, tpr, thresholds = metrics.roc_curve(testing_predict_alreadygiven, testing_df_predicted_values_randomforest, pos_label=1)
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

print "Gini:"
print "Confusion Matrix: "
print confusion_matrix(testing_predict_alreadygiven, testing_df_predicted_values)
print "Accuracy: " , accuracy_score(testing_predict_alreadygiven, testing_df_predicted_values)
print "F1 Score: " , f1_score(testing_predict_alreadygiven, testing_df_predicted_values, average="binary")
print "Precision: " , precision_score(testing_predict_alreadygiven, testing_df_predicted_values, average="binary")
print "Recall: " , recall_score(testing_predict_alreadygiven, testing_df_predicted_values, average="binary")

print "-------------------------------------------------------------------------------------------------------------------------------"

print "Entropy:"
print "Confusion Matrix: "
print confusion_matrix(testing_predict_alreadygiven, testing_df_perdicted_values_entropy)

print "Accuracy: " , accuracy_score(testing_predict_alreadygiven, testing_df_perdicted_values_entropy)
print "F1 Score: " , f1_score(testing_predict_alreadygiven, testing_df_perdicted_values_entropy, average="binary")
print "Precision: " ,precision_score(testing_predict_alreadygiven, testing_df_perdicted_values_entropy, average="binary")
print "Recall: " , recall_score(testing_predict_alreadygiven, testing_df_perdicted_values_entropy, average="binary")

print "-------------------------------------------------------------------------------------------------------------------------------"

print "Random Forest Classifier:"
print "Confusion Matrix: "
print confusion_matrix(testing_predict_alreadygiven, testing_df_predicted_values_randomforest)

print "Accuracy: " , accuracy_score(testing_predict_alreadygiven, testing_df_predicted_values_randomforest)
print "F1 Score: " , f1_score(testing_predict_alreadygiven, testing_df_predicted_values_randomforest, average="binary")
print "Precision: " ,precision_score(testing_predict_alreadygiven, testing_df_predicted_values_randomforest, average="binary")
print "Recall: " , recall_score(testing_predict_alreadygiven, testing_df_predicted_values_randomforest, average="binary")