from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['marital','nonmarital','single'])
a=list(le.transform(['marital','nonmarital','single','marital','nonmarital','single']))
print(a)
le.fit(['play','not play'])
b=list(le.transform(['play','not play','play','not play','play','not play']))
print(b)