"""CS512
    Aurelio Arango
    Kristina Nystrom
    Marshia Hashemi"""


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




#------------------------------------------------------------------------------------

print "Program is now running..."
start = time.time()


model = GA.GA()
fileW = FromFitnessFileGA.createAnOutputFile()
numOfPop = 50
numOfFea = 396
num_ga_pop = 20
numberOfGenerations = 2000

"""Read data from file"""
TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileGA.getAllOfTheData()
"""Rescale the data!!! """
TrainX, ValidateX, TestX = FromDataFileGA.rescaleTheData(TrainX, ValidateX, TestX)

"""Create population, random"""
population = model.Create_A_Population(numOfPop, numOfFea)

fittingStatus, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, \
        trackR2PredTest = FromFitnessFileGA.validate_model(model, fileW, population, TrainX, TrainY, \
                                                           ValidateX, ValidateY, TestX, TestY)

"""Variables that will keep track of the parents fitness and population"""
parentsPopulation = zeros((2, numOfFea))
parentsFitness = [0] * 2
"""Sorting population to get the best fitness"""
model.sort_population(population, fitness, numOfPop)

parentsPopulation[0] = population[0]
parentsPopulation[1] = population[1]
parentsFitness[0] = fitness[0]
parentsFitness[1] = fitness[1]
"""Variable that keeps track of the last time fitness was updated, (changes)"""
fitnessLastUpdated = 0

"""For loops goes through many generations up to 2000 and it checks every 500 if no changes were made"""
for x in range(0, numberOfGenerations):
    population = model.Create_GA_Population(numOfPop, numOfFea, population, num_ga_pop, fitness)


    fittingStatus, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, \
            trackR2PredTest = FromFitnessFileGA.validate_model(model, fileW, population, TrainX, TrainY, \
                                                           ValidateX, ValidateY, TestX, TestY)
    """Sort population"""
    model.sort_population(population, fitness, numOfPop)
    """"""
    if ((parentsFitness[0] > fitness[0]) or (parentsFitness[1] > fitness[1])):
        parentsPopulation[0] = population[0]
        parentsPopulation[1] = population[1]
        parentsFitness[0] = fitness[0]
        parentsFitness[1] = fitness[1]
        fitnessLastUpdated = 0
    if (fitnessLastUpdated == 500):
        # print fitnessLastUpdated
        break
    fitnessLastUpdated = fitnessLastUpdated + 1
print "\nGenerations: " + str(numberOfGenerations)
print "Fitness last updated: " + str(fitnessLastUpdated)
end = time.time()
print str((end - start) / 60) + " minutes"
"""Saving Data to file"""
FromFitnessFileGA.write(model, fileW, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, trackR2PredTest)





