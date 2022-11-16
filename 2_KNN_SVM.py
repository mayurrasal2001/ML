import pandas as pd
import numpy as np
df = pd.read_csv("emails.csv")
print(df)

print(df.shape)

df.drop(['Email No.'],axis=1, inplace=True)
print(df.shape)

x = df.drop(['Prediction'],axis=1)
y = df['Prediction']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.4,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, mean_squared_error
y_predict = classifier.predict(x_test)

print(accuracy_score(y_test, y_predict))

print(mean_squared_error(y_test, y_predict))

import math
print(math.sqrt(mean_squared_error(y_test, y_predict)))