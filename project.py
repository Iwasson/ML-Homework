import numpy as np
from sklearn.metrics import confusion_matrix

HIDDENUNITS = 175
HIDDEN2 = 150
HIDDEN3 = 125
HIDDEN4 = 100
HIDDEN5 = 75
LEARNINGRATE = 0.01
EPOCHS = 1
MOMENTUM = 0.9
BIAS = 1

dataset1 = np.load('project_Dataset/data_batch_1', allow_pickle=True, encoding='bytes')
dataset2 = np.load('project_Dataset/data_batch_2', allow_pickle=True, encoding='bytes')
dataset3 = np.load('project_Dataset/data_batch_3', allow_pickle=True, encoding='bytes')
dataset4 = np.load('project_Dataset/data_batch_4', allow_pickle=True, encoding='bytes')
dataset5 = np.load('project_Dataset/data_batch_5', allow_pickle=True, encoding='bytes')
testset = np.load('project_Dataset/test_batch', allow_pickle=True, encoding='bytes')

trainData = np.concatenate((dataset1[b'data'], dataset2[b'data'], dataset3[b'data'], dataset4[b'data'], dataset5[b'data']), axis=0)
trainLabels = np.concatenate((dataset1[b'labels'], dataset2[b'labels'], dataset3[b'labels'], dataset4[b'labels'], dataset5[b'labels']), axis=0)
trainData = trainData / 255

testData = testset[b'data']
testData = testData / 255
testLabels = testset[b'labels']

testingConfusionLabels = []
testingConfusionResults = []

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

def runEpoch(learningRate, momentum, training, inToHidden, hiddenTo2, hiddenTo3, hiddenTo4, hiddenTo5, hiddenToOut, prevIn, prev2, prev3, prev4, prev5, prevOut):
    totalCorrect = 0
    
    for i in range(0, len(trainData)):
        inputs = trainData[i] 
        inputs = np.insert(inputs, [0], [BIAS], axis=0)
        inputs = inputs.reshape(1, len(inputs)) 
        label = trainLabels[i]

        # Forward Propagation
        hiddenInputs = np.dot(inputs, inToHidden)
        hiddenOutputs = sigmoid(hiddenInputs)
        
        hidden2Inputs = np.dot(hiddenOutputs, hiddenTo2)
        hidden2Outputs = sigmoid(hidden2Inputs)

        hidden3Inputs = np.dot(hidden2Outputs, hiddenTo3)
        hidden3Outputs = sigmoid(hidden3Inputs)

        hidden4Inputs = np.dot(hidden3Outputs, hiddenTo4)
        hidden4Outputs = sigmoid(hidden4Inputs)

        hidden5Inputs = np.dot(hidden4Outputs, hiddenTo5)
        hidden5Outputs = sigmoid(hidden5Inputs)

        outInputs = np.dot(hidden5Outputs, hiddenToOut)
        outputs = sigmoid(outInputs)
        #outputs = outputs.reshape(1, len(outputs[0]))

        # Backward Propagation
        targetOutputs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        targetOutputs[label] = 0.9

        prediction = np.argmax(outputs) 
        if prediction == label:
            totalCorrect += 1
        if training:
            outputError = outputs * (1 - outputs) * (targetOutputs - outputs)
            hidden5Error = hidden5Outputs * (1 - hidden5Outputs) * np.dot(outputError, np.transpose(hiddenToOut))
            hidden4Error = hidden4Outputs * (1 - hidden4Outputs) * np.dot(hidden5Error, np.transpose(hiddenTo5))
            hidden3Error = hidden3Outputs * (1 - hidden3Outputs) * np.dot(hidden4Error, np.transpose(hiddenTo4))
            hidden2Error = hidden2Outputs * (1 - hidden2Outputs) * np.dot(hidden3Error, np.transpose(hiddenTo3))
            hiddenError = hiddenOutputs * (1 - hiddenOutputs) * np.dot(hidden2Error, np.transpose(hiddenTo2))
            #inError = inputs * (1 - inputs) * np.dot(hiddenError, np.transpose(inToHidden))

            # Update Weights
            deltaInputToHidden = (learningRate * np.dot(np.transpose(inputs), hiddenError)) + (momentum * prevIn)
            deltaHiddenTo2 = (learningRate * np.dot(np.transpose(hiddenOutputs), hidden2Error)) + (momentum * prev2)
            deltaHiddenTo3 = (learningRate * np.dot(np.transpose(hidden2Outputs), hidden3Error)) + (momentum * prev3)
            deltaHiddenTo4 = (learningRate * np.dot(np.transpose(hidden3Outputs), hidden4Error)) + (momentum * prev4)
            deltaHiddenTo5 = (learningRate * np.dot(np.transpose(hidden4Outputs), hidden5Error)) + (momentum * prev5)
            deltaHiddenToOut = (learningRate * np.dot(np.transpose(hidden5Outputs), outputError)) + (momentum * prevOut)

            inToHidden += deltaInputToHidden
            prevIn = deltaInputToHidden
            hiddenTo2 += deltaHiddenTo2
            prev2 = deltaHiddenTo2
            hiddenTo3 += deltaHiddenTo3
            prev3 = deltaHiddenTo3
            hiddenTo4 += deltaHiddenTo4
            prev4 = deltaHiddenTo4
            hiddenTo5 += deltaHiddenTo5
            prev5 = deltaHiddenTo5
            hiddenToOut += deltaHiddenToOut
            prevOut = deltaHiddenToOut
        
        else:
            testingConfusionLabels.append(label)
            testingConfusionResults.append(prediction)
    
    return (totalCorrect / len(trainData)) * 100, inToHidden, hiddenTo2, hiddenTo3, hiddenTo4, hiddenTo5, hiddenToOut, prevIn, prev2, prev3, prev4, prev5, prevOut

def experiment1():
    #inToHidden = randWeightMatrix(3073, HIDDENUNITS + 1)
    #hiddenToOut = randWeightMatrix(HIDDENUNITS + 1, 10)
    #global inToHidden, hiddenTo2, hiddenTo3, hiddenTo4, hiddenTo5, hiddenToOut
    #global prevIn, prevOut, prev2, prev3, prev4, prev5
    inToHidden = randWeightMatrix(3073, HIDDENUNITS + 1)
    hiddenTo2 = randWeightMatrix(HIDDENUNITS + 1, HIDDEN2 + 1)
    hiddenTo3 = randWeightMatrix(HIDDEN2 + 1, HIDDEN3 + 1)
    hiddenTo4 = randWeightMatrix(HIDDEN3 + 1, HIDDEN4 + 1)
    hiddenTo5 = randWeightMatrix(HIDDEN4 + 1, HIDDEN5 + 1)
    hiddenToOut = randWeightMatrix(HIDDEN5 + 1, 10)

    prevIn = np.zeros((1, HIDDENUNITS + 1))
    prev2 = np.zeros((1, HIDDEN2 + 1))
    prev3 = np.zeros((1, HIDDEN3 + 1))
    prev4 = np.zeros((1, HIDDEN4 + 1))
    prev5 = np.zeros((1, HIDDEN5 + 1))
    prevOut = np.zeros((1, 10))
    
    for i in range(0, EPOCHS):
        accuracy, inToHidden, hiddenTo2, hiddenTo3, hiddenTo4, hiddenTo5, hiddenToOut, prevIn, prev2, prev3, prev4, prev5, prevOut = runEpoch(LEARNINGRATE, MOMENTUM, True, inToHidden, hiddenTo2, hiddenTo3, hiddenTo4, hiddenTo5, hiddenToOut, prevIn, prev2, prev3, prev4, prev5, prevOut)
        print("Epoch: " + str(i) + " Accuracy: " + str(accuracy))
    print(confusion_matrix(testingConfusionLabels, testingConfusionResults))

experiment1()