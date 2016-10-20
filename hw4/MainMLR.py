
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

def main():


    fileW = FromFinessFileMLR.createAnOutputFile()
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
    population = DifferentialEvolution.Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = FromFinessFileMLR.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

#main program ends in here

#------------------------------------------------------------------------------

main()
#------------------------------------------------------------------------------



