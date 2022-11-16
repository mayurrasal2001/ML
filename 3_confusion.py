import pandas as pd
import numpy as np
df = pd.read_csv("diabetes.csv")
print(df)

print(df.shape)

x = df.drop(['Outcome'],axis=1)
y = df['Outcome']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
y_predict = classifier.predict(x_test)

print(accuracy_score(y_test, y_predict))

print(confusion_matrix(y_test, y_predict))

tp,fp,fn,tn = confusion_matrix(y_test, y_predict).ravel()
print(tp)
print(fp)
print(fn)
print(tn)

accuracy = (tp + tn)*100/ (tp+tn+fp+fn)
print("Accuracy")
print(accuracy)

precious = tp / tp + fp
print("Precious")
print(precious)

recall = tp / tp + fn
print("Recall")
print(recall)

error = (fp+fn)/(tp+tn+fp+fn)
print("Error")
print(error)