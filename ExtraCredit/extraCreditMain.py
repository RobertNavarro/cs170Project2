import pandas as pd
import numpy as np
import numba as nb
import scipy
from numba import jit, njit, prange
import random
import time
import sys

########################################-Forward Selection-######################################################
@njit(parallel=True)
def leaveOneOutCrossValidation(dataToModify, currentSet, featureToAdd):
    modifiedSet = np.copy(currentSet) #copies the list so the new feature can be added without changing the original list
    if(featureToAdd != -1):
        modifiedSet = np.append(modifiedSet, featureToAdd)
    for feature in range(1,array_size2):
        if feature not in modifiedSet:
            dataToModify[:, feature] = 0

    numberOfCorrectlyClassified = 0
    for row in prange(array_size1):
        objectToClassify = dataToModify[row, 1:] #all columns except the label column
        labelOfObjectToClassify = dataToModify[row, 0] #label
        nearestNeighborDistance = np.Infinity
        nearestNeighborLocation = np.Infinity
        nearestNeighborLabel = -1
        for k in prange(array_size1):
            if k != row: #only if the two objects are not the same object
                otherObject = dataToModify[k, 1:]#all columns except the label column
                #distance = np.linalg.norm(objectToClassify-otherObject)#euclidean distance
                distance = np.sqrt(np.sum(np.square(objectToClassify-otherObject)))
                if distance < nearestNeighborDistance:
                    nearestNeighborDistance = distance
                    nearestNeighborLocation = k
                    nearestNeighborLabel = dataToModify[nearestNeighborLocation, 0]
        if labelOfObjectToClassify == nearestNeighborLabel: #everytime we get a match increase the count
            numberOfCorrectlyClassified += 1
        #print("Object " + str(row) + " is class " + str(labelOfObjectToClassify))   
        #print("It's nearest neighbor is " + str(nearestNeighborLocation) + " which is in class " + str(nearestNeighborLabel))
    accuracy = numberOfCorrectlyClassified / array_size1
    if featureToAdd != -1: #just for aesthetic purposes when printing
        print(accuracy)
    return accuracy

def featureSearch():
    bestFeatures = []
    currentSetOfFeatures = np.copy(bestFeatures)
    allTimeBestAccuracy = leaveOneOutCrossValidation(np.ndarray.copy(data), currentSetOfFeatures, -1)
    print("Running nearest neighbor with no features, using \"leaving-one-out\" evaluation, I get an accuracy of", str(allTimeBestAccuracy*100) + "%" )
    print("Beginning Search")
    for col in range(1, array_size2):
        featureToAddAtThisLevel = -1
        bestSoFarAccuracy = 0
        print("on the " + str(col) + "th level of the search tree")
        for feature in range(1, array_size2):
            if feature not in currentSetOfFeatures:
                print("--Consider adding the " + str(feature))
                dataToModify =  np.ndarray.copy(data)
                accuracy = leaveOneOutCrossValidation(dataToModify, currentSetOfFeatures, feature)
                #accuracy = random.uniform(0, 1)

                if accuracy > bestSoFarAccuracy:
                    bestSoFarAccuracy = accuracy
                    featureToAddAtThisLevel = feature
        if featureToAddAtThisLevel != -1:
            currentSetOfFeatures = np.append(currentSetOfFeatures,featureToAddAtThisLevel)
        print("on level " + str(col) + " I added feature " + str(featureToAddAtThisLevel) + " and now the current set is", currentSetOfFeatures.astype(int))
        #allTimeBestAccuracy = max(allTimeBestAccuracy, bestSoFarAccuracy)
        if allTimeBestAccuracy < bestSoFarAccuracy:
            allTimeBestAccuracy = bestSoFarAccuracy
            bestFeatures = np.copy(currentSetOfFeatures)
    '''#uncomment for testing
    f = open("output.txt", "a")
    accuracyOutput = ("The best accuracy was" ,allTimeBestAccuracy)
    featureOutput = ("The best features were" , bestFeatures.astype(int))
    f.write(str(accuracyOutput)+"\n")
    f.write(str(featureOutput)+"\n")
    f.close()
    '''
    print("The best accuracy was" ,allTimeBestAccuracy)
    print("The best features were" , bestFeatures.astype(int))


########################################-Backward Elimination-######################################################
@njit(parallel=True)
def leaveOneOutCrossValidationBackwards(dataToModify, currentSet, featureToRemove):
    modifiedSet = np.copy(currentSet) #copies the list so the new feature can be added without changing the original list
    if(featureToRemove != -1):
        deletionIndex = np.argwhere(modifiedSet == featureToRemove)
        modifiedSet = np.delete(modifiedSet, deletionIndex[0])
    for feature in range(1,array_size2):
        if feature not in modifiedSet:
            dataToModify[:, feature] = 0

    numberOfCorrectlyClassified = 0
    for row in prange(array_size1):
        objectToClassify = dataToModify[row, 1:] #all columns except the label column
        labelOfObjectToClassify = dataToModify[row, 0] #label
        nearestNeighborDistance = np.Infinity
        nearestNeighborLocation = np.Infinity
        nearestNeighborLabel = -1
        for k in prange(array_size1):
            if k != row: #only if the two objects are not the same object
                otherObject = dataToModify[k, 1:]#all columns except the label column
                #distance = np.linalg.norm(objectToClassify-otherObject)#euclidean distance
                distance = np.sqrt(np.sum(np.square(objectToClassify-otherObject)))
                if distance < nearestNeighborDistance:
                    nearestNeighborDistance = distance
                    nearestNeighborLocation = k
                    nearestNeighborLabel = dataToModify[nearestNeighborLocation, 0]
        if labelOfObjectToClassify == nearestNeighborLabel: #everytime we get a match increase the count
            numberOfCorrectlyClassified += 1
        #print("Object " + str(row) + " is class " + str(labelOfObjectToClassify))   
        #print("It's nearest neighbor is " + str(nearestNeighborLocation) + " which is in class " + str(nearestNeighborLabel))
    accuracy = numberOfCorrectlyClassified / array_size1
    if featureToRemove != -1: #just for aesthetic purposes when printing
        print(accuracy)
    return accuracy

def backwardfeatureSearch():
    bestFeatures = np.array(list(range(1,array_size2)))
    currentSetOfFeatures = np.array(list(range(1,array_size2)))
    allTimeBestAccuracy = leaveOneOutCrossValidationBackwards(np.ndarray.copy(data), currentSetOfFeatures, -1)
    print("Running nearest neighbor with all features, using \"leaving-one-out\" evaluation, I get an accuracy of", str(allTimeBestAccuracy*100) + "%")
    print("Beginning Search")
    for col in range(1, array_size2):
        featureToRemoveAtThisLevel = -1
        bestSoFarAccuracy = 0
        print("on the " + str(col) + "th level of the search tree")
        for feature in range(1, array_size2):
            if feature in currentSetOfFeatures:
                print("--Consider removing the " + str(feature) + " feature from the set")
                dataToModify =  np.ndarray.copy(data)
                accuracy = leaveOneOutCrossValidationBackwards(dataToModify, currentSetOfFeatures, feature)
                #accuracy = random.uniform(0, 1)

                if accuracy > bestSoFarAccuracy:
                    bestSoFarAccuracy = accuracy
                    featureToRemoveAtThisLevel = feature
        if featureToRemoveAtThisLevel != -1:
            #currentSetOfFeatures.delete(featureToRemoveAtThisLevel)
            currentSetOfFeatures = np.delete(currentSetOfFeatures, np.argwhere(currentSetOfFeatures == featureToRemoveAtThisLevel))
        print("on level " + str(col) + " I removed feature " + str(featureToRemoveAtThisLevel) + " and now the current set is", currentSetOfFeatures)
        if allTimeBestAccuracy < bestSoFarAccuracy:
            allTimeBestAccuracy = bestSoFarAccuracy
            bestFeatures = np.copy(currentSetOfFeatures)
    '''#uncomment for testing
    f = open("output.txt", "a")
    accuracyOutput = ("The best accuracy was" ,allTimeBestAccuracy)
    featureOutput = ("The best features were" , bestFeatures.astype(int))
    f.write(str(accuracyOutput)+"\n")
    f.write(str(featureOutput)+"\n")
    f.close()
    '''
    print("The best accuracy was" ,allTimeBestAccuracy)
    print("The best features were" , bestFeatures.astype(int))
########################################################################################
def main():
    #print(array_size2)
    #featureSearch()
    global data
    global array_size1
    global array_size2
    print("Welcome to Robert Navarro's Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ")#sys.argv[1]
    print("filename",filename)
    data = pd.read_csv(filename, header=None,engine='python').to_numpy()
        
    array_size1= (np.shape(data))[0] #number of rows aka the value of K
    array_size2 = (np.shape(data))[1] #number of columns
    for column in range(1,array_size2):
        data[:, column] = (data[:, column] -  np.mean(data[:, column])) / np.std(data[:, column])
    startTime = 0
    algorithm = input("\nType the number of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n")#sys.argv[2]
    if algorithm == "1":
        startTime = time.time()
        featureSearch()
    elif algorithm == "2":
        startTime = time.time()
        backwardfeatureSearch()
    '''#uncomment for testing
    f = open("output.txt", "a")
    if algorithm == "1":
        timeOutput = (round((time.time() - startTime),2), "seconds to run algorithm on", filename, "using forward selection")
    elif algorithm == "2":
        timeOutput = (round((time.time() - startTime),2), "seconds to run algorithm on", filename, "using backward elimination")
    f.write(str(timeOutput)+"\n"+"------------------------------------------------------------------------------------------------------------------------------\n\n")
    f.close()    
    '''
    print(round((time.time() - startTime),2), "seconds to run algorithm on", filename)
    return 0
main()