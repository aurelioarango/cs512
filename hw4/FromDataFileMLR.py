import time                 # provides timing for benchmarks
from numpy  import *        # provides complex math and array functions
from sklearn import svm	    # provides Support Vector Regression
import csv
import math
import sys


#------------------------------------------------------------------------------
def getTwoDecPoint(x):
    return float("%.2f"%x)

#------------------------------------------------------------------------------
def placeDataIntoArray(fileName):
    with open(fileName, mode='rbU') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        dataArray = array([row for row in datareader], dtype=float64, order='C')

    if (min(dataArray.shape) == 1): # flatten arrays of one row or column
        return dataArray.flatten(order='C')
    else:
        return dataArray
        
#------------------------------------------------------------------------------
def getAllOfTheData():
    TrainX = placeDataIntoArray('Train-Data.csv')
    TrainY = placeDataIntoArray('Train-pIC50.csv')
    ValidateX = placeDataIntoArray('Validation-Data.csv')
    ValidateY = placeDataIntoArray('Validation-pIC50.csv')
    TestX = placeDataIntoArray('Test-Data.csv')
    TestY = placeDataIntoArray('Test-pIC50.csv')
    return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY

#------------------------------------------------------------------------------

def rescaleTheData(TrainX, ValidateX, TestX):

    # 1 degree of freedom means (ddof) N-1 unbiased estimation
    TrainXVar = TrainX.var(axis = 0, ddof=1)  # get the variance
    TrainXMean = TrainX.mean(axis = 0)  # get the mean

    for i in range(0, TrainX.shape[0]):
        TrainX[i,:] = (TrainX[i,:] - TrainXMean)/sqrt(TrainXVar)
    for i in range(0, ValidateX.shape[0]):
        ValidateX[i,:] = (ValidateX[i,:] - TrainXMean)/sqrt(TrainXVar)
    for i in range(0, TestX.shape[0]):
        TestX[i,:] = (TestX[i,:] - TrainXMean)/sqrt(TrainXVar)

    return TrainX, ValidateX, TestX

#------------------------------------------------------------------------------
