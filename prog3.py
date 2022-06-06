import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('hw3_Dataset/data.txt')
iteration = 0

def selectK(dataset, k):
    indices = np.random.choice(dataset.shape[0], k, replace=False)
    return dataset[indices]

def euclideanDistance(x, y):
    return np.sqrt(np.sum((x-y)**2))


# create an array for each k value
# compute the euclidean distance for each point in the dataset against each k value
# push the smallest distance to the correspoding k value array
# compute the mean of the k value array
# update the k value based off of that mean
# repeat

def assignKarray(dataset, k_values):
    k_arrays = [[] for i in range(len(k_values))]

    # loop through the dataset and compute the euclidean distance
    # then push the smallest distance to the correspoding k value array
    for i in range(len(dataset)):
        distances = []
        for j in range(len(k_values)):
            distances.append(euclideanDistance(dataset[i], k_values[j]))
        
        minK = np.argmin(distances) # index of the smallest distance
        k_arrays[minK].append(dataset[i])
        
    return k_arrays


def computeMeans(k_array):
    return np.mean(k_array, axis=0)

def plotter(kCount, clusters, kValues):
    global iteration
    for i in range(kCount): #Plots a multi demensional graph of the clusters
        temp = np.asarray(clusters[i])
        plt.scatter(temp.T[0],temp.T[1], cmap=plt.get_cmap(name=None, lut=None)) #Assigning colors to the clusters
        plt.scatter(kValues.T[0], kValues.T[1], c='black')
    plt.savefig('plot' + str(iteration) + '.png')
    iteration += 1 
    return


def kMeans(k, r):
    # select the k values that we want to start with
    k_values = selectK(dataset, k)
    k_arrays = []

    for episode in range(r):
        k_arrays = assignKarray(dataset, k_values)

        plotter(k, k_arrays, k_values)

        for i in range(k):
            k_values[i] = computeMeans(k_arrays[i])
    
    final_ks = assignKarray(dataset, k_values)
    squares = 0
    for i in range(k):
        for j in range(len(final_ks[i])):
            squares += np.square(final_ks[i][j] - k_values[i])
    plotter(k, final_ks, k_values)

    return k_values, np.sum(squares)


print(kMeans(5, 5))