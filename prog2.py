import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys

def genDataSets():
    dataset = np.loadtxt('./hw2_Dataset/spambase.data', delimiter=',')
    dataTrain, dataTest, LabelTrain, labelTest = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.5)
    return dataTrain, dataTest, LabelTrain, labelTest

def probModel(label):
    totalSpam = (np.count_nonzero(label) / len(label))
    totalNotSpam = 1 - totalSpam
    return totalSpam, totalNotSpam

def stdANDmean(data, label):
    totalSpam = []
    totalNotSpam = []
    mean = np.ones((2, 57))
    std = np.ones((2, 57))

    for i in range(len(data)):
        if label[i] == 1:
            totalSpam.append(data[i])
        else:
            totalNotSpam.append(data[i])
    
    totalSpam = np.array(totalSpam)
    totalNotSpam = np.array(totalNotSpam)

    for i in range(57):
        mean[0][i] = np.mean(totalNotSpam[:, i])
        mean[1][i] = np.mean(totalSpam[:, i])
        std[0][i] = np.std(totalNotSpam[:, i])
        std[1][i] = np.std(totalSpam.T[i])
    std[std == 0] = 0.001

    return mean, std

def argMax(probSpam, probNotSpam):
    if probSpam > probNotSpam:
        return 1
    else:
        return 0

def bayesProb(trainData, trainLabel, testData):
    mean, std = stdANDmean(trainData, trainLabel)
    totalSpam, totalNotSpam = probModel(trainLabel)

    probSpam = 0
    probNotSpam = 0

    classes = []

    for x in range(len(testData)):
        for i in range(2):
            probability = 0
            if i == 0:
                probability = np.log(totalNotSpam)
            else:
                probability = np.log(totalSpam)
            
            for j in range(57):
                tempProb = (1 / (std[i][j] * np.sqrt(2 * np.pi))) * np.exp(-((testData[x][j] - mean[i][j]) ** 2) / (2 * std[i][j] ** 2))
                if tempProb == 0:
                    tempProb = sys.float_info.min
                probability += np.log(tempProb)
            
            if i == 0:
                probNotSpam = probability
            else:
                probSpam = probability
        
        classes.append(argMax(probSpam, probNotSpam))
    return classes

def main():
    dataTrain, dataTest, labelTrain, labelTest = genDataSets()
    results = bayesProb(dataTrain, labelTrain, dataTest)
    accuracy = 0
    truePositive = 0
    falseNegative = 0
    falsePositive = 0

    for i in range(len(results)):
        if results[i] == labelTest[i]:
            accuracy += 1
        if results[i] == 1 and labelTest[i] == 1:
            truePositive += 1
        if results[i] == 0 and labelTest[i] == 1:
            falseNegative += 1
        if results[i] == 1 and labelTest[i] == 0:
            falsePositive += 1
    
    print("Accuracy: ", accuracy / len(results))
    print("Recall: ", truePositive / (truePositive + falseNegative))
    print("Percision: ", truePositive / (truePositive + falsePositive))
    print("Confusion Matrix: \n", confusion_matrix(labelTest, results))

main()