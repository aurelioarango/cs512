
import time                 #provides timing for benchmarks
from numpy import *         #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv
import math
import sys

#Local files created by me
import mlr
import FromDataFileGA
import FromFitnessFileGA
				
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
    #print population[0]

    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]              
    return population
#------------------------------------------------------------------------------
def sort_population(population,fitness, numOfPop):
    """sort matrix by fitness: We will use a quick sort"""
    """This quick sort will sort the fitness array from lowest to highest.
        Additionally, at the same time it will sort its corresponding population """
    quick_sort_population(population,fitness,0,numOfPop)

def quick_sort_population(population,fitness,iLow,iHigh ):
    pivotpoint=0
    if(iHigh>iLow):
        pivotpoint = sort_partition(population,fitness,iLow,iHigh)
        quick_sort_population(population,fitness,iLow,pivotpoint)
        quick_sort_population(population,fitness,pivotpoint+1,iHigh)

def sort_partition(population,fitness,iLow,iHigh):
    """Partition of quick sort """
    #i=iLow+1
    j=iLow
    temp=0
    pivot_item= fitness[iLow]

    for i in range(iLow+1,iHigh):
        if(fitness[i] < pivot_item):
            j = j+1
            temp = fitness[i]
            fitness[i] = fitness[j]
            fitness[j] = temp
            temp_pop = population[i]
            population[i] = population[j]
            population[j] = temp_pop
    pivot_point = j
    temp = fitness[iLow]
    fitness[iLow] = fitness[pivot_point]
    fitness[pivot_point] = temp
    temp_pop = population[iLow]
    population[iLow] = population[pivot_point]
    population[pivot_point] = temp_pop

    return pivot_point
#------------------------------------------------------------------------------
def split_parents(mom,dad,numSplitPoints,numOfFeatures):
    """Split the parents by the number of split points given by the user"""
    """Check that the new child has at lest 3 features """
    child_one_sum=0
    child_two_sum=0

    while(child_one_sum < 3) and (child_two_sum <3):
        splitpoint = random.randint(0, numOfFeatures)

        child_one = concatenate((mom[0:splitpoint], dad[splitpoint:]))
        child_two = concatenate((dad[0:splitpoint], mom[splitpoint:]))
        child_one_sum = child_one.sum()
        child_two_sum = child_two.sum()

    return child_one, child_two

""" -------------------------UNDER CONSTRUCTION---------------------"""
def random_split(num_splits, numOfFeatures):
    """returns and array with the number of split points"""

    split_points = zeros(num_splits)
    for x in range(0, num_splits):
        split_points[x] = random.randint(0, numOfFeatures)
    """SORT NUMBERS BEFORE IMPLEMENTING"""
    return split_points
def split_parents_helper(mom,dad,splitpoints,num_splits):
    if(num_splits == 1):
        child_one = concatenate((mom[0:splitpoints[0]], dad[splitpoints[0]:]))
        child_two = concatenate((dad[0:splitpoints[0]], mom[splitpoints[0]:]))
    #elif(num_splits==2):

    #elif (num_splits == 3):

    #elif (num_splits == 4):

    #elif (num_splits == 5):
"""------------------------------------------------------------------"""
#------------------------------------------------------------------------------
def Create_GA_Population(numOfPop,numOfFea,previousPop,numSplitPoints, fitness  ):
    """get the previous population"""
    """Sort population by fitness using a quick sort """

    sort_population(previousPop,fitness,numOfPop)


    
    ga_population=20
    new_pop = zeros((ga_population, numOfFea))
    j =0
    """Get the best two parents"""
    new_pop[0] = previousPop[0]
    new_pop[1] = previousPop[1]
    
    """split the best forloop to create children best of 10 add to new population"""
    for x in range(2,ga_population,2):
        new_pop[x], new_pop[x+1] = split_parents(previousPop[j],previousPop[j+1],numSplitPoints,numOfFea)
        j=j+1

    """Create 30 new random samples calling the function Create_A_Population"""
    random_population = Create_A_Population(numOfPop - ga_population, numOfFea)
    
    new_pop = concatenate((new_pop,random_population))

    print shape(new_pop)

    """return the new population set"""
    return new_pop

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


    #fileW = createAnOutputFile()
    fileW=0;
    model = mlr.MLR()

#Number of descriptor should be 396 and number of population should be 50 or more

    numOfPop = 50
    numOfFea = 396
    unfit = 1000
    splitpoint = 0

# Final model requirements

    R2req_train    = .6
    R2req_validate = .5
    R2req_test     = .5
    """Read data from file"""
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileGA.getAllOfTheData()
    """Rescale the data!!! """
    TrainX, ValidateX, TestX = FromDataFileGA.rescaleTheData(TrainX, ValidateX, TestX)

    unfit = 1000
    fittingStatus = unfit
    """Create a population based on the number of features selected, in this case 10, from the pool of features"""
    """Initial population"""
    population = Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = FromFitnessFileGA.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    """Create loop that will generate at most 2000 generations """
    """Check every 500 generations for changes in population if no change print to file"""
    
    """Create GA population, based on first population"""
    population = Create_GA_Population (numOfPop,numOfFea,population,splitpoint, fitness)
    """ """
    FromFitnessFileGA.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    #print population
    #print str(fittingStatus) +" "+str(fitness)
#main program ends in here

#------------------------------------------------------------------------------

main()
#------------------------------------------------------------------------------



