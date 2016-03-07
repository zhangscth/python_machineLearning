from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier(n_neighbors=10) #10nn

iris = datasets.load_iris()

print(iris)
knn.fit(iris.data,iris.target)

print("\n")
print("knn : "+str(knn))

predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])

print(str(predictedLabel))