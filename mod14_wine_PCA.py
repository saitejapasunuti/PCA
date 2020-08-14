############ DIMENSION REDUCTION PCA ############

import pandas as pd
#Pandas is used for data manipulation, analysis,cleaning
import numpy as np
#it deals with numerical data
wine=pd.read_csv("D:/360digiTMG/unsupervised/mod14 PCA/wine.csv")
wine

wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#data visualization
from sklearn.preprocessing import scale
#for standardizing values

#normalizing the numerical values
wine_norm=scale(wine)
wine_norm

pca=PCA(n_components=14)
pca_values=pca.fit_transform(wine_norm)

#the amount of variance each pca explains is
var=pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]

var1=np.cumsum(np.round(var,decimals=4)*100)
var1

#variance plot for PCA components obtained
plt.plot(var1,color="red")

#plot between pca1 and pca2
x=pca_values[:,0]
y=pca_values[:,1]
z=pca_values[:,2]

pca_values

################ hierarchical clustering ################

#performing hirarchical clustering using 3 principle component scores
data=pca_values[:,0:3]
data

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
#it is used for the creating dendrograms

z=linkage(data,method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
        z,
        leaf_rotation=0.,
        leaf_font_size=8.,
        )
plt.show()
#based on the dendrogram we take clusters value =3

from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(data)

clusters_labels=pd.Series(h_complete.labels_)




############## KMeans ##################

from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 

data_k=pca_values[:,0:3]
data_k

k=list(range(1,10))
k

TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data_k)
    WSS=[]#var for storing within sum of sqrs
    for j in range(i):
        WSS.append(sum(cdist(data_k[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,data_k.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
#based on the elbow curve wee can consider k value as 3

model=KMeans(n_clusters=3)
model.fit(data_k)

model.labels_# getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)
md

