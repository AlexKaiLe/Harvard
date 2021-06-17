import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import average
# import pandas as pd
# import scipy.cluster.hierarchy as shc
# from sklearn.cluster import AgglomerativeClustering

# # %matplotlib inline
# import numpy as np
# def clusterPlot(data):
#     cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
#     cluster.fit_predict(data)
#     plt.figure(figsize=(10, 7))
#     plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
#     plt.show()

# customer_data = pd.read_csv('data.csv')
# # print(customer_data.shape)
# # print(customer_data.head())
# data = customer_data.iloc[:, 3:5].values
# print(data)
# print(type(data))
# clusterPlot(data)

# # plt.figure(figsize=(10, 7))
# # plt.title("Customer Dendograms")
# # dend = shc.dendrogram(shc.linkage(data, method='ward'))
# # plt.show()

import seaborn as sns

# x1,y1 = np.random.normal(loc=0.0, scale=1.0, size=(100,)), np.random.normal(loc=2.0, scale=1.0, size=(100,))
# x2,y2 = np.random.normal(loc=2., scale=1.0, size=(100,)), np.random.normal(loc=0.0, scale=1.0, size=(100,))
x1 = np.array([126.50128559629657, 107.7481555370784, 144.47172763032344, 124.71698916173688, 111.45307193617549, 137.5665672855291, 148.09501875190148, 129.06145957080412])
y1 = np.array([95.87751097003226, 94.60340820326826, 100.96289783926407, 108.422923415146, 116.69024998999717, 106.39083209949477, 113.87919153824915, 124.4882107401799])
x2 = np.array([54.43781552320668, 74.1892349459479, 79.66850095053866, 69.98713752703672, 59.45611965374812])
y2 = np.array([111.01353123085968, 91.45283065698186, 102.59988335011819, 112.62546649450576, 127.99257481250064])
x3 = np.array([0.0])
y3 = np.array([0.0])
x4 = np.array([100.98564418950552, 103.67269868481499, 89.81696226547697, 94.72881429121561])
y4 = np.array([137.5737876942737, 142.42263985108073, 145.88746155927407, 153.40558387737997])
x5 = np.array([19.88343585660168, 37.037628755952994])
y5 = np.array([80.43306363109302, 70.1948307214227])




fig, ax = plt.subplots()
# sns.displot(x=x1, y=y1, kind="kde", color='r')
# sns.displot(x=x2, y=y2, kind="kde", color='b')
sns.kdeplot(x = x1, y = y1, shade=False, cmap='Greys')
sns.kdeplot(x = x2, y = y2, shade=True, cbar=False, ax=ax, cmap='Greys')
# sns.kdeplot(x = x3, y = y3, shade=True, alpha=0.3, cbar=False, ax=ax, cmap="Blues")
sns.kdeplot(x = x4, y = y4, shade=True, cbar=False, ax=ax, cmap='Greys')
# sns.kdeplot(x = x5, y = y5, shade=True, alpha=0.3, cbar=False, ax=ax, cmap="Oranges")

plt.scatter(x1,y1, color="C0")
plt.scatter(x2,y2, color="C1")
plt.scatter(x3,y3, color="C1")
plt.scatter(x4,y4, color="C1")
plt.scatter(x5,y5, color="C1")
plt.show()