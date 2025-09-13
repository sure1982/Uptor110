import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_excel('tripDetails.xlsx')

print(data)
print (data.head())

data.drop(['TripID'],axis = 1,inplace = True)
print (data.head())

features = list(data.columns)
print(features)

units = ['kms','kmph','kmph','mins','counts','mins','counts']
feature_units = dict(zip(features,units))
print (feature_units)

data.info()

print (data.describe())

for item in features:
    data[item].plot(kind='hist', bins = 15)
    plt.title(item)
    plt.xlabel(feature_units[item])
    plt.show()

correlation = data.corr()
print(correlation)

sns.heatmap(np.abs(correlation), xticklabels = correlation.columns, yticklabels = correlation)
plt.show()

sns.pairplot(data)
plt.show()

from sklearn.preprocessing import StandardScaler

data2 = data.copy()
data2 = StandardScaler().fit_transform(data2.values)
data2 = pd.DataFrame(data2,columns = features)

from sklearn import cluster

distortions = [] # Empty list to store wss
for i in range(1, 11):
    km = cluster.KMeans(n_clusters=i,
            init='k-means++',
            n_init = 10,
            max_iter = 300,
            random_state = 100)
    km.fit(data2.values)
    distortions.append(km.inertia_)
#Plotting the K-means Elbow plot
plt.figure(figsize = (7,7))
plt.plot(range(1,11), distortions, marker='o')
plt.title('ELBOW PLOT')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

k = 3
km3 = cluster.KMeans(n_clusters=k,
                init='k-means++',
                n_init = 10,
                max_iter = 300,
                random_state = 100)
print (km3.fit(data2.values))

labels = km3.labels_
Ccenters = km3.cluster_centers_
data2['labels'] = labels
data2['labels'] = data2['labels'].astype('str')
print(data2['labels'])

sns.pairplot(data2, x_vars = features, y_vars = features, hue='labels', diag_kind='kde')
plt.show()

c_df = pd.concat([data[data2['labels']=='0'].mean(),
data[data2['labels']=='1'].mean(),
data[data2['labels']=='2'].mean()],
axis=1)
c_df.columns = ['cluster1','cluster2','cluster3']
c_df

triptype = ['Intercity-Peak hours','Highway','Intercity-Non-peak hours']
data['labels'] = labels
data['labels'] = data['labels'].map({0:triptype[0],1:triptype[1],2:triptype[2]})

print(data.head())
