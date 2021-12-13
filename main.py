# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
from csv import reader
from math import sqrt
import random
from typing import Mapping


def loadData():
	temp = []
	uniqueClass = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
	print('Class lables Transform to integers as,\n', uniqueClass, '\n', '-'*100)
	file = open('iris.data', 'r')
	for line in file.readlines():
		line = line.strip()
		tline = line.split(',')
		# print(tline)
		temp.append([float(tline[0]), float(tline[1]), float(tline[2]), float(tline[3]), uniqueClass[tline[4]]])
	file.close()
	return temp

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(4):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def get_prediction(trainset, testpoint, num_neighbors):
	distances = []
	for trainpoint in trainset:
		dist = euclidean_distance(trainpoint, testpoint)
		distances.append((trainpoint[-1], dist))
	distances.sort(key=lambda x: x[1])
	neighborsclass = []
	for i in range(num_neighbors):
		neighborsclass.append(distances[i][0])
	predictedclass = max(set(neighborsclass), key=neighborsclass.count)
	return predictedclass

def KNN(trainset, testset, num_neighbors):
	predictedclass = []
	for testpoint in testset:
		predictedclass.append(get_prediction(trainset, testpoint, num_neighbors))
	return predictedclass

def normalizeDataset(dataset):
	minT = [1<<10]*4
	maxT = [0]*4
	for trainpoint in dataset:
		for i, x in enumerate(trainpoint[:-1]):
			minT[i] = min(minT[i], x)
			maxT[i] = max(maxT[i], x)
	

	for i, trainpoint in enumerate(dataset):
		for j, x in enumerate(trainpoint[:-1]):
			dataset[i][j] = (x - minT[j])/(maxT[j] - minT[j])
	return dataset


def printDataset(dataset, s):
	print()
	print('-'*100)
	print(s, '( count =', len(dataset), ') :')
	print('-'*100)
	print('sepal length\tsepal width\tpetal length\tpetal width\tclass')
	print('-'*100)
	for x in dataset:
		print('%1.2f\t\t%1.2f\t\t%1.2f\t\t%1.2f\t\t%d' %(x[0], x[1], x[2], x[3], x[4]))
	print('-'*100)

#loading dataset
dataset = loadData()
printDataset(dataset, 'Inicial dataset')

#normalizing dataset
dataset = normalizeDataset(dataset)
printDataset(dataset, 'Dataset after normalization')

#slipting dataset into train and test in ratio 70:30
random.shuffle(dataset)
trainset = dataset[:int(len(dataset)*.7)]
testset = dataset[len(trainset):]
printDataset(trainset, 'Train dataset')
printDataset(testset, 'Test dataset')

#defining parameter
countNeighbours = 5

#real class of test data
y_test = []
for i in testset:
	y_test.append(i[-1])

#predicted class of test data using KNN
y_pred = KNN(trainset, testset, countNeighbours)


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print('Confusion Matrix : \n', confusion_matrix(y_test, y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred)*100, "%")
print(classification_report(y_test, y_pred))