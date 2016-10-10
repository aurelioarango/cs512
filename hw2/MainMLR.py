
import time                 #provides timing for benchmarks
from numpy  import *        #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv
import math
import sys

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR
				
#------------------------------------------------------------------------------
def getAValidrow(numOfFea, eps=0.015):
    sum = 0
    while (sum < 3):
       V = zeros(numOfFea)
       for j in range(numOfFea):
          r = random.uniform(0,1)
          if (r < eps):
             V[j] = 1
          else:
             V[j] = 0
       sum = V.sum()
    return V
#------------------------------------------------------------------------------

def Create_A_Population(numOfPop, numOfFea):
    population = random.random((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]              
    return population

#------------------------------------------------------------------------------
def createAnOutputFile():

    file_name = None
    algorithm = None


    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if ( (file_name == None) and (algorithm != None)):
        file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
                        alg.model.__class__.__name__, alg.gen_max,timestamp)
    elif file_name==None:
        file_name = "{}.csv".format(timestamp)
    fileOut = file(file_name, 'wb')
    fileW = csv.writer(fileOut)

    fileW.writerow(['Descriptor ID', 'Fitness', 'Model','R2', 'Q2', \
            'R2Pred_Validation', 'R2Pred_Test'])

    return fileW

#------------------------------------------------------------------------------

def main():


    fileW = createAnOutputFile()
    model = mlr.MLR()

#Number of descriptor should be 396 and number of population should be 50 or more

    numOfPop = 50
    numOfFea = 396
    unfit = 1000

# Final model requirements

    R2req_train    = .6
    R2req_validate = .5
    R2req_test     = .5

    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    unfit = 1000
    fittingStatus = unfit
    """Create a population based on the number of features selected, in this case 10, from the pool of features"""
    population = Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = FromFinessFileMLR.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

#main program ends in here

#------------------------------------------------------------------------------

main()
#------------------------------------------------------------------------------



