import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys



import GA
import Sort
import FromDataFileGA
import FromFitnessFileGA



def createAnOutputFile():
    file_name = None
    algorithm = None

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if ((file_name == None) and (algorithm != None)):
        file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
                                                alg.model.__class__.__name__, alg.gen_max, timestamp)
    elif file_name == None:
        file_name = "{}.csv".format(timestamp)
    fileOut = file(file_name, 'wb')
    fileW = csv.writer(fileOut)

    fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', \
                    'R2Pred_Validation', 'R2Pred_Test'])

    return fileW


#------------------------------------------------------------------------------------
model = GA.GA()
fileW = createAnOutputFile()
numOfPop = 50
numOfFea = 396
num_ga_pop = 20

"""Read data from file"""
TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileGA.getAllOfTheData()
"""Rescale the data!!! """
TrainX, ValidateX, TestX = FromDataFileGA.rescaleTheData(TrainX, ValidateX, TestX)

"""Create population, random"""
population = model.Create_A_Population(numOfPop, numOfFea)

fittingStatus, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, \
        trackR2PredTest = FromFitnessFileGA.validate_model(model, fileW, population, TrainX, TrainY, \
                                                           ValidateX, ValidateY, TestX, TestY)


population = model.Create_GA_Population(numOfPop, numOfFea, population, num_ga_pop, fitness)

"""Read data from file"""
#TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileGA.getAllOfTheData()
"""Rescale the data!!! """
#TrainX, ValidateX, TestX = FromDataFileGA.rescaleTheData(TrainX, ValidateX, TestX)

#data = {itFits, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, trackR2PredTest}
fittingStatus, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, \
        trackR2PredTest = FromFitnessFileGA.validate_model(model, fileW, population, TrainX, TrainY, \
                                                           ValidateX, ValidateY, TestX, TestY)


#print data
FromFitnessFileGA.write(model, fileW, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, trackR2PredTest)

#FromFitnessFileGA.write(model, fileW, data[2], data[3], data[4], data[5], data[6], data[7], data[8])





