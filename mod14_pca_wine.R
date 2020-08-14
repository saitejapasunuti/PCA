################## Dimenson Reduction ###########################

#load wine dataframe

wine <- read.csv(file.choose())
View(wine)

attach(wine)

pcaObj <- princomp(wine,cor = TRUE,scores = TRUE,covmat = NULL)
str(pcaObj)
pcaObj

summary(pcaObj)

loadings(pcaObj)

plot(pcaObj)# graph showing importance of principal components

biplot(pcaObj)

plot(cumsum(pcaObj$sdev*pcaObj$sdev)*100/sum(pcaObj$sdev*pcaObj$sdev),type="b")


pcaObj$scores
pcaObj$scores[,1:3]

final <- cbind(wine[,1],pcaObj$scores[,1:3])
View(final)

##########hierarchical clustering for the first three principle components

install.packages("plyr")
library(plyr)
#plyr package is used for manipulation

install.packages("animation")
library(animation)
#it is used for the visualization purpose

install.packages("kselection")
library(kselection)

data <- pcaObj$scores[,1:3]
d <- dist(data,method = "euclidean")#distance matrix
fit <- hclust(d,method = "complete")

plot(fit)

plot(fit,hang = -1)
groups <- cutree(fit,k=3)# taking 4 has clusters based on the dendrogram
rect.hclust(fit,k=3,border = "blue")

membership <- as.matrix(groups)

final1 <- data.frame(data,membership)
View(final1)

final2 <- final1[,c(ncol(final1),1:(ncol(final1)-1))]
View(final2)


##k means clustering 

install.packages("kselection")
library(kselection)

k <- kselection(data[,-5], parallel = TRUE, k_threshold = 0.9, )
k
#f(k) finds 3 clusters

install.packages("doParallel")
library(doParallel)
registerDoParallel(cores=3)
k <- kselection(data, parallel = TRUE, k_threshold = 0.9, max_centers=8)
k

fit <- kmeans(data, 3) # 3 cluster solution
str(fit)
fit
final3<- data.frame(data, fit$cluster) # append cluster membership
final3
final4 <- final3[,c(ncol(final3),1:(ncol(final3)-1))]
aggregate(data, by=list(fit$cluster), FUN=mean)

#elbow curve & k ~ sqrt(n/2) to decide the k value

wss = (nrow(data)-1)*sum(apply(data, 2, var))		 # Determine number of clusters by scree-plot 
for (i in 1:8) wss[i] = sum(kmeans(data, centers=i)$withinss)
plot(1:8, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")   # Look for an "elbow" in the scree plot #
title(sub = "K-Means Clustering Scree-Plot")

#based on the elbow curve we can take the k value as 3

km <- kmeans(data,3)
km

#Within cluster sum of squares by cluster:
#[1] 224.0527 169.2931  97.8848
#(between_SS / total_SS =  70.9 %)

#K-means clustering with 3 clusters of sizes 67, 62, 49

