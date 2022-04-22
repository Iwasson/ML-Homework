import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix
from os.path import exists

learningRate1 = 0.001
learningRate2 = 0.01
learningRate3 = 0.1

allWeights = []
testingConfusionLabels = []
testingConfusionResults = []

dataJson1 = "data.1.json"
dataJson01 = "data.01.json"
dataJson001 = "data.001.json"

MAXEPOCHS = 70

trainingFile = pd.read_csv('hw1_Dataset/mnist_train.csv')
testFile = pd.read_csv('hw1_Dataset/mnist_test.csv')

convertedTrainingData = trainingFile.to_numpy()
convertedTestData = testFile.to_numpy()

def randWeightT():
    weight = []
    weight = np.random.uniform(-0.05, 0.05, 785)
    return weight

#file_exists = exists(dataJson001)
#if file_exists:
#    loadedFile = open(dataJson001, 'r')
#    jsonFile = json.load(loadedFile)
#    allWeights = np.array(jsonFile)
#    allWeights = np.reshape(allWeights, (10, 785))
#    print("LOADING")
#else:
#   print("MAKING NEW MATRIX")
#   for i in range(0,10):
#       allWeights.append(randWeightT())


#create the 785 inputs for the perceptron
#scale the data down to be between 0 and 1
#do this by dividing each value by 255
def loadImage(row, training):
    inputs = []
    if training:
        convertedData = convertedTrainingData
    else:
        convertedData = convertedTestData

    for i in range(0,784):
        inputs.append((convertedData[row][i+1])/255)
    inputs.append(1)
    return inputs

#load the label for the given row, this is the number that the image is representing
def loadLabel(row, training):
    label = 0
    if training:
        convertedData = convertedTrainingData
    else:
        convertedData = convertedTestData
    
    label = convertedData[row][0]
    return label
    

#creates a neuron from the weights and the inputs to get our predictions
def makePredicitons(inputs):
    predictions = []
    for x in range(0,10):
        neuron = 0
        for i in range(0,785):
            neuron += allWeights[x][i] * inputs[i]
        predictions.append(neuron)
    return predictions


def updateWeights(learningRate, inputs, predictions, label):
    for x in range(0,10):
        for i in range(0,785):
            y = 0
            t = 0
            #if(x == label and predictions[x] > 0):
            if(x == label):
                t = 1
            if(predictions[x] > 0):
                y = 1
            
            allWeights[x][i] = allWeights[x][i] + (learningRate * (t-y) * inputs[i])
            
    #saveData = np.array(allWeights)
    #with open('data.json', 'w') as f:
        #json.dump(saveData.tolist(), f, ensure_ascii=False)
        #f.write(jsonDump)

#run a single epoch, output the prediction as well as the accuracy of this epoch
def runEpoch(learningRate, training, dataJson):
    totalRows = 0       #how many rows are present in the dataset
    totalCorrect = 0    #the total number of correct perdictions, compute accuracy by dividing this by totalRows

    #select which dataset we are looking at
    if training:
        totalRows = len(trainingFile.index) - 1
        
    else:
        totalRows = len(testFile.index) - 1

    #run through the dataset
    for row in range(0, totalRows):
        inputs = loadImage(row+1, training)            #load the image
        predictions = makePredicitons(inputs)   #make the prediction
        label = loadLabel(row+1, training)              #load the label


        finalPrediction = 0
        for i in range(0,10):
            if predictions[i] > predictions[finalPrediction]:
                finalPrediction = i
        
        if training == False:
            testingConfusionResults.append(finalPrediction)
            testingConfusionLabels.append(label)
        #if the perceptron made the correct guess, increment the totalCorrect
        if finalPrediction == label:
            totalCorrect += 1
        #otherwise, update the weights, only if the training is set to true
        elif finalPrediction != label and training:
            updateWeights(learningRate, inputs, predictions, label)
    if training:
        saveData = np.array(allWeights)
        with open(dataJson, 'w') as f:
            json.dump(saveData.tolist(), f, ensure_ascii=False)
            #f.write(jsonDump)

    #return the accuracy of the epoch
    return (totalCorrect/totalRows) * 100


def runTraining(learningRate, i, dataJson):
    accuracy = runEpoch(learningRate, True, dataJson)
    print("Training Epoch: " + str(i) + " Accuracy: " + str(accuracy))

def runTesting(learningRate, i, dataJson):
    accuracy = runEpoch(learningRate, False, dataJson)
    print("Testing Epoch: " + str(i) + " Accuracy: " + str(accuracy))
    
def runProgram(allWeights, testingConfusionLabels, testingConfusionResults):
    print("Learning Rate: 0.001")
    file_exists = exists(dataJson001)
    if file_exists:
        loadedFile = open(dataJson001, 'r')
        jsonFile = json.load(loadedFile)
        allWeights = np.array(jsonFile)
        allWeights = np.reshape(allWeights, (10, 785))
        print("LOADING")
    else:
        print("MAKING NEW MATRIX")
        for i in range(0,10):
            allWeights.append(randWeightT())

    for i in range(0, MAXEPOCHS):
        runTraining(learningRate1, i, dataJson001)
        runTesting(learningRate1, i, dataJson001)
    matrix = confusion_matrix(testingConfusionLabels, testingConfusionResults)
    print(matrix)

    print("===============================================================")
    print("Learning Rate: 0.01")
    allWeights.clear()
    testingConfusionLabels.clear()
    testingConfusionResults.clear()
    file_exists = exists(dataJson01)
    if file_exists:
        loadedFile = open(dataJson01, 'r')
        jsonFile = json.load(loadedFile)
        allWeights = np.array(jsonFile)
        allWeights = np.reshape(allWeights, (10, 785))
        print("LOADING")
    else:
        print("MAKING NEW MATRIX")
        for i in range(0,10):
            allWeights.append(randWeightT())
    for i in range(0, MAXEPOCHS):
        runTraining(learningRate2, i, dataJson01)
        runTesting(learningRate2, i, dataJson01)
    matrix = confusion_matrix(testingConfusionLabels, testingConfusionResults)
    print(matrix)

    print("===============================================================")
    print("Learning Rate: 0.1")
    allWeights.clear()
    testingConfusionLabels.clear()
    testingConfusionResults.clear()
    file_exists = exists(dataJson1)
    if file_exists:
        loadedFile = open(dataJson1, 'r')
        jsonFile = json.load(loadedFile)
        allWeights = np.array(jsonFile)
        allWeights = np.reshape(allWeights, (10, 785))
        print("LOADING")
    else:
        print("MAKING NEW MATRIX")
        for i in range(0,10):
            allWeights.append(randWeightT())
    for i in range(0, MAXEPOCHS):
        runTraining(learningRate3, i, dataJson1)
        runTesting(learningRate3, i, dataJson1)
    matrix = confusion_matrix(testingConfusionLabels, testingConfusionResults)
    print(matrix)
    
runProgram(allWeights, testingConfusionLabels, testingConfusionResults)