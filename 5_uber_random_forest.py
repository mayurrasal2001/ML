import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\ganes\Downloads\ml\New folder\uber.csv")
df.head()
df.info()
df = df.drop(['Unnamed: 0', 'key'], axis =1)
df.head()
df.isna().sum()
df = df.dropna(axis=0)
df.isna().sum()
df.shape
df.dtypes
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df.assign(hour = df.pickup_datetime.dt.hour,
              day = df.pickup_datetime.dt.day,
              month = df.pickup_datetime.dt.month,
              year = df.pickup_datetime.dt.year,
              dayofweek = df.pickup_datetime.dt.day_of_week)
df = df.drop("pickup_datetime", axis =1)

from sklearn.model_selection import train_test_split
x = df.drop('fare_amount', axis=1)
y=df.fare_amount
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print("\nMAE:{:.2f}".format(mae))
print("\nMSE:{:.2f}".format(mse))
print("\nRMSE:{:.2f}".format(rmse))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print("\nMAE:{:.2f}".format(mae))
print("\nMSE:{:.2f}".format(mse))
print("\nRMSE:{:.2f}".format(rmse))



RangeIndex: 200000 entries, 0 to 199999
Data columns (total 9 columns):
 #   Column             Non-Null Count   Dtype  
---  ------             --------------   -----  
 0   Unnamed: 0         200000 non-null  int64  
 1   key                200000 non-null  object 
 2   fare_amount        200000 non-null  float64
 3   pickup_datetime    200000 non-null  object 
 4   pickup_longitude   200000 non-null  float64
 5   pickup_latitude    200000 non-null  float64
 6   dropoff_longitude  199999 non-null  float64
 7   dropoff_latitude   199999 non-null  float64
 8   passenger_count    200000 non-null  int64  
dtypes: float64(5), int64(2), object(2)
memory usage: 13.7+ MB

MAE:5.99

MSE:102.29

RMSE:10.11