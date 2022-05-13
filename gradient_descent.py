import numpy as np
from numpy.linalg import norm

def gradX(x):
    #return 4 * x - 4
    return 4 * x * x * x

def gradY(y):
    #return 6 * y + 24
    return 2 * y

def randVal():
    return np.random.uniform(-10, 10)

def gradientDescent(maxSteps, learningRate):
    x = randVal()
    y = randVal()
    previousX = x
    previousY = y
    converged = False
    for i in range(maxSteps):
        if(i > 0 and not converged and norm(np.array([x,y]) - np.array([previousX, previousY])) < 0.00001):
            print(norm(np.array([x,y]) - np.array([previousX, previousY])))
            converged = True
            print("Converged at iteration: " + str(i))
        previousX = x
        previousY = y
        x -= learningRate * gradX(x)
        y -= learningRate * gradY(y)
    return x, y

print(gradientDescent(500000, 0.01))