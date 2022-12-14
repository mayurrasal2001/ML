import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("sales_data_sample.csv",sep=",", encoding='Latin-1')
df.info()
x = df.iloc[:,[1,4]].values
x
from sklearn.cluster import KMeans
#elbow method
wcss_list = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', random_state=12)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

plt.plot(range(1,11), wcss_list)
plt.title("Elbow Method Graph")
plt.xlabel('Number of clusters (k)')
plt.ylabel('wcss_list')
plt.show()

# K-mean clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=23)
y_pred = kmeans.fit_predict(x)

plt.scatter(x[y_pred==0,0], x[y_pred==0,1], c='blue',label='cluster 1')
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], c='red',label='cluster 2')
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], c='green',label='cluster 3')
plt.title('K-means Clustering')
plt.xlabel('Quantity Ordered')
plt.ylabel('Sales')
plt.legend()
plt.show()