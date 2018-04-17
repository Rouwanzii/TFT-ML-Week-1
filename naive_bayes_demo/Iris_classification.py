import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv("./Iris.csv")
print(iris.shape)
print(iris.head(3))

le = LabelEncoder()
le.fit(iris['Species'])
iris['Species'] = le.transform(iris['Species']) # Transform Categorie Species Into Integers
print("Categories in Iris: ", list(le.classes_))

# Split the dataset into 2/3 training data and 1/3 test data
trainSet, testSet = train_test_split(iris, test_size = 0.33)
print(trainSet.shape)
print(testSet.shape)
print(trainSet.head(3))

# Training and testing data process
trainData = pd.DataFrame.as_matrix(trainSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
trainTarget = pd.DataFrame.as_matrix(trainSet[['Species']]).ravel()
testData = pd.DataFrame.as_matrix(testSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
testTarget = pd.DataFrame.as_matrix(testSet[['Species']]).ravel()

classifier = GaussianNB()
classifier.fit(trainData, trainTarget)

predictedValues = classifier.predict(testData)

nErrors = (testTarget != predictedValues).sum()
accuracy = 1.0 - nErrors / testTarget.shape[0]
print("Accuracy: ", accuracy)

