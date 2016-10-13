import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys

# Local files created by me
import GA
import FromDataFileGA
import FromFitnessFileGA
import Sort


# ------------------------------------------------------------------------------
def getAValidrow(numOfFea, eps=0.015):
    sum = 0
    while (sum < 3):
        V = zeros(numOfFea)
        for j in range(numOfFea):
            r = random.uniform(0, 1)
            if (r < eps):
                V[j] = 1
            else:
                V[j] = 0
        sum = V.sum()
    return V


# ------------------------------------------------------------------------------

def Create_A_Population(numOfPop, numOfFea):
    population = random.random((numOfPop, numOfFea))
    # print population[0]

    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population

def sort_population(population, fitness, numOfPop):
    """sort matrix by fitness: We will use a quick sort"""
    """This quick sort will sort the fitness array from lowest to highest.
        Additionally, at the same time it will sort its corresponding population """
    # print numOfPop
    Sort.quick_sort_population(population, fitness, 0, numOfPop)


# ------------------------------------------------------------------------------
def split_parents(mom, dad, numOfFeatures):
    """Split the parents by the number of split points given by the user"""
    """Check that the new child has at lest 3 features """
    child_one_sum = 0  # Initialize variables to 0
    child_two_sum = 0

    while (child_one_sum < 3) and (child_two_sum < 3):
        splitpoint = random.randint(0, numOfFeatures)
        '''found that we did not need to seed the random because we did get
           different values every time we printed it'''
        child_one = concatenate((mom[0:splitpoint], dad[splitpoint:]))
        child_two = concatenate((dad[0:splitpoint], mom[splitpoint:]))
        child_one_sum = child_one.sum()
        child_two_sum = child_two.sum()
    """Return the two children"""
    return child_one, child_two

# ------------------------------------------------------------------------------
def Create_GA_Population(numOfPop, numOfFea, previousPop, ga_pop, fitness):
    """get the previous population"""
    """Sort population by fitness using a quick sort """
    print str(len(fitness)) + " fitness ga"
    sort_population(previousPop, fitness, numOfPop)


    """ Genetic algorithm population size"""
    ga_population = ga_pop

    new_pop = zeros((ga_population, numOfFea))
    print shape(new_pop)

    if ga_population > 2:
        #new_pop = zeros((ga_population, numOfFea))
        j = 0
        """Get the best two parents"""
        new_pop[0] = previousPop[0]
        new_pop[1] = previousPop[1]
        #print new_pop[0]
        #print new_pop[1]

        """split the best for loop to create children best of 10 add to new population"""
        """from 2 till n, it will """
        counter = 2;
        print ga_population

        for x in range(2, ga_population, 2):
            new_pop[x], new_pop[x + 1] = split_parents(previousPop[x], previousPop[x + 1], numOfFea)

        #for x in range(2, ga_population,2):
        #    new_pop[x], new_pop[x + 1] = split_parents(previousPop[j], previousPop[j + 1], numOfFea)
         #   print str(x) +" x "
         #   j = j + 1
         #   counter =counter +2;
        #print str(counter) + " counter - number of children with parents"
    """Create new random samples calling the function Create_A_Population """

    print str(len(fitness)) + " fitness ga before random pop"
    print str(numOfPop - ga_population)+ " numOfPop - ga_population"
    print str(shape(new_pop)) + " ga population"

    random_population = Create_A_Population(numOfPop - ga_population, numOfFea)

    print str(shape(random_population)) + "random population"

    if ga_population >= 2:
        new_pop = concatenate((new_pop, random_population))
        print str(shape(new_pop)) + " inside if ga"
    else:
        print str(shape(random_population))+" inside else ga"
        new_pop = random_population
        return new_pop

    print str(len(fitness)) + " fitness ga after if"

    """return the new population set"""
    return new_pop


# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------

def main():
    print "Program is now running..."
    start = time.time()

    fileW = createAnOutputFile()
    #fileW = 0;
    model = GA.GA()

    # Number of descriptor should be 396 and number of population should be 50 or more

    numOfPop = 50
    numOfFea = 396
    unfit = 1000
    splitpoint = 0

    # Final model requirements

    R2req_train = .6
    R2req_validate = .5
    R2req_test = .5
    """Read data from file"""
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileGA.getAllOfTheData()
    """Rescale the data!!! """
    TrainX, ValidateX, TestX = FromDataFileGA.rescaleTheData(TrainX, ValidateX, TestX)

    unfit = 1000
    fittingStatus = unfit
    """Create a population based on the number of features selected, in this case 10, from the pool of features"""
    """Initial population"""
    population = Create_A_Population(numOfPop, numOfFea)



    fittingStatus, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, \
trackR2PredTest = FromFitnessFileGA.validate_model(model, fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    """Create loop that will generate at most 2000 generations """
    numberOfGenerations = 10
    parentsPopulation = zeros((2, numOfFea))
    parentsFitness = [0]*2
    num_ga_pop = 20
    sort_population(population, fitness, numOfPop)

    """keep track of best parents throughout the generations"""
    parentsPopulation[0] = population[0]
    parentsPopulation[1] = population[1]
    parentsFitness[0] = fitness[0]
    parentsFitness[1] = fitness[1]
    fitnessLastUpdated = 0

    for x in range(0, numberOfGenerations):

        population = Create_A_Population(numOfPop, numOfFea)
        """Create GA population, based on first population"""

        population = Create_GA_Population(numOfPop, numOfFea, population, num_ga_pop, fitness)
        """Validate model"""
        #print str(shape(population)) + " pop shape"

        #print "-----------before validation loop "+ str(len(fitness))
        fittingStatus, fitness, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, \
        trackR2PredTest = FromFitnessFileGA.validate_model(model, fileW, population, \
            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


        #print str(shape(population)) + " after validation pop shape"
        print "----------- after validation loop"+ str(len(fitness))
        sort_population(population, fitness, numOfPop)
        print "----------- after sort loop" + str(len(fitness))

        print str(shape(population)) + " after sort pop shape"
        """changes the best populations and fitnesses if the top two populations are better - lower"""
        if ((parentsFitness[0] > fitness[0]) or (parentsFitness[1] > fitness[1])):
            parentsPopulation[0] = population[0]
            parentsPopulation[1] = population[1]
            parentsFitness[0] = fitness[0]
            parentsFitness[1] = fitness[1]
            fitnessLastUpdated = 0

        """Check every 500 generations for changes in population if no change print to file"""
        """checks to make sure the best fitness and populations changed within 500 iterations"""
        if(fitnessLastUpdated == 500):
            #print fitnessLastUpdated
            break

        fitnessLastUpdated = fitnessLastUpdated + 1
    print fitness
    print "----------- before writing to file" + str(len(fitness))

    print str(shape(population)) + " before writing to file pop shape"
    FromFitnessFileGA.write(model, fileW, trackDesc, trackFitness, trackModel, trackR2, trackQ2, trackR2PredValidation, trackR2PredTest)

    """tells us how much time has elapsed since we started"""
    end = time.time()
    print str((end - start)/60) + " minutes"




    """ """

    """Need to keep track of which model was used to gather features"""
    print shape(population)
    # print str(fittingStatus) +" "+str(fitness)


# main program ends in here

# ------------------------------------------------------------------------------

main()
# ------------------------------------------------------------------------------



