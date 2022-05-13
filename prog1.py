import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from os.path import exists

trainingFile = pd.read_csv('hw1_Dataset/mnist_train.csv')
testFile = pd.read_csv('hw1_Dataset/mnist_test.csv')

MAXEPOCHS = 50

hiddenUnits = 100
learningRate = 0.1
bias = 1
epochs = 50
momentum = 0.9

testingConfusionLabels = []
testingConfusionResults = []

convertedTrainingData = trainingFile.to_numpy()
convertedTestData = testFile.to_numpy()


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig
    
def randWeights(num):
    weight = []
    weight = np.random.uniform(-0.05, 0.05, num)
    return weight

def randWeightMatrix(n,m):
    weightMatrix = []
    for i in range(0,n):
        weightMatrix.append(randWeights(m))
    return weightMatrix

def loadImage(row, training):
    inputs = []
    if training:
        convertedData = convertedTrainingData
    else:
        convertedData = convertedTestData

    for i in range(0,784):
        inputs.append((convertedData[row][i+1])/255)
    return inputs

def loadLabel(row, training):
    label = 0
    if training:
        convertedData = convertedTrainingData
    else:
        convertedData = convertedTestData
    
    label = convertedData[row][0]
    return label

def runEpoch(training, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, startRows, endRows, momentum):
    #totalRows = 0
    totalCorrect = 0

    #if training:
    #    totalRows = len(trainingFile.index) - 1
    #else:
    #    totalRows = len(testFile.index) - 1
    
    #for row in range(totalRows):
    #for row in range(0, (totalRows / 2)):
    for row in range(startRows, endRows):
        inputs = loadImage(row, training)                               #stores all of the pixel values
        inputs = np.insert(inputs, [0], [bias], axis=0)                 #adds the bias to the inputs
        inputs = inputs.reshape(1,785)                                  #reshapes the inputs to be a row vector
        label = loadLabel(row, training)                                #stores the expected prediction value for this row

        hiddenInputs = np.dot(inputs, inputToHiddenWeights)             #perform dot product on the inputs and the input weights
        hiddenInputs = sigmoid(hiddenInputs)                            #apply sigmoid function to the hidden inputs

        output = np.dot(hiddenInputs, hiddenToOutputLayerWeights)       #perform dot product on the hidden inputs and the hidden weights
        output = sigmoid(output)                                        #apply sigmoid function to the output

        
        targetOutputs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        targetOutputs[label] = 0.9

        prediction = np.argmax(output)                                  #find the index of the highest value in the output
        if prediction == label:
            totalCorrect += 1
        if training:
            #update weights by first calculating the error, do this working from the output layer to the hidden layer then the hidden layer to the input layer
            outputError = output * (1 - output) * (targetOutputs - output)                                              #calculate the error for the output layer
            hiddenError = hiddenInputs * (1 - hiddenInputs) * np.dot(outputError, np.transpose(hiddenToOutputLayerWeights))

            #calculate the deltaInput
            deltaInput = (learningRate * hiddenError * (inputs.T)) + (momentum * previousInputWeights)
            previousInputWeights = deltaInput
            inputToHiddenWeights = inputToHiddenWeights + deltaInput

            #calculate the deltaHidden
            deltaHidden = (learningRate * (outputError * np.transpose(hiddenInputs))) + (momentum * previousHiddenLayerWeights)
            previousHiddenLayerWeights = deltaHidden
            hiddenToOutputLayerWeights = hiddenToOutputLayerWeights + deltaHidden
            
        else:
            testingConfusionResults.append(prediction)
            testingConfusionLabels.append(label) 

    return (totalCorrect / (endRows - startRows)) * 100, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights


#inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
#hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)
#
#previousInputWeights = np.zeros((785, hiddenUnits + 1))
#previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

#print("Training...")
#print(runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1)))


#vary the hidden units
def experiment1():
    global testingConfusionLabels, testingConfusionResults
    hiddenUnits = 20
    momentum = 0.9
    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    print("Experiment 1")
    print("20 Hidden units")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1), momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(testFile.index) - 1), momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    hiddenUnits = 50
    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    print("50 Hidden units")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1), momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(testFile.index) - 1), momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    hiddenUnits = 100
    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    print("100 Hidden units")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1), momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(testFile.index) - 1), momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))
    return 0

#vary the momentum
def experiment2():
    global testingConfusionLabels, testingConfusionResults
    hiddenUnits = 100
    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    print("Experiment 2")
    momentum = 0
    print("0.1 momentum")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1), momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(testFile.index) - 1), momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    momentum = 0.25
    print("0.25 momentum")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1), momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(testFile.index) - 1), momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    momentum = 0.5
    print("0.5 momentum")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(trainingFile.index) - 1), momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, (len(testFile.index) - 1), momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))
    
    return 0

#vary the sample size
def experiment3():
    global testingConfusionLabels, testingConfusionResults
    momentum = 0.9
    hiddenUnits = 100
    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    print("Experiment 3")
    print("Half samples")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, int(len(trainingFile.index) / 2) - 1, momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, len(testFile.index) - 1, momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    print("Quarter samples")
    for i in range(MAXEPOCHS):
        print("Epoch: " + str(i))
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(True, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, int(len(trainingFile.index) / 4) - 1, momentum)
        print("Training: ",accuracy)
        accuracy, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights = runEpoch(False, inputToHiddenWeights, hiddenToOutputLayerWeights, previousInputWeights, previousHiddenLayerWeights, 0, len(testFile.index) - 1, momentum)
        print("Testing: ",accuracy)
        print("")
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))
    testingConfusionLabels = []
    testingConfusionResults = []

    inputToHiddenWeights = randWeightMatrix(785, hiddenUnits + 1)
    hiddenToOutputLayerWeights = randWeightMatrix(hiddenUnits + 1, 10)

    previousInputWeights = np.zeros((785, hiddenUnits + 1))
    previousHiddenLayerWeights = np.zeros((hiddenUnits + 1, 10))

    return 0

experiment1()
experiment2()
experiment3()