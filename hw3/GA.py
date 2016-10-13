"""CS512
    Aurelio Arango
    Kristina Nystrom
    Marshia Hashemi"""
"""Genetic Algorithm"""

import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys


import numpy as np
import Sort

class GA:
    """Genetic Algorithm"""
    def __init__(self):
        """Initialization"""
        self.coef = None

    def fit(self, x_set, y_set):
        """Fit to training X and Y arrays"""
        # Add a column of 1's for the intercept
        x_set = np.append(np.ones((x_set.shape[0], 1)), x_set, axis=1)
        """ returns the least squares solution for the population R or Q, """
        self.coef = np.linalg.lstsq(x_set, y_set)[0]
        return 'GA'

    def predict(self, x_set):
        """Predict a Y from an X, object must already be fitted."""
        # matrix multiplication of X appended with a column of 1's (for
        # intercepts) and the coeficients
        """Check if the len of the matrix is 1"""
        if len(x_set.shape) == 1:
            """Change the shape of the array without changing the data"""
            """goes from a single array to a matrix"""
            x_set = np.reshape(x_set, (1, x_set.shape[0]))
            """Add the rest of the data"""
        x_set = np.append(np.ones((x_set.shape[0], 1)), x_set, axis=1)
        """Returns the matrix multiplication"""
        return np.dot(x_set, self.coef)

    def printing(self):
        """Predict a Y from an X, object must already be fitted."""
        print "How are you doing?"

    def getAValidrow(self,numOfFea, eps=0.015):
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

    def Create_A_Population(self,numOfPop, numOfFea):
        population = random.random((numOfPop, numOfFea))

        for i in range(numOfPop):
            V = self.getAValidrow(numOfFea)
            for j in range(numOfFea):
                population[i][j] = V[j]
        return population

    def sort_population(self,population, fitness, numOfPop):
        """sort matrix by fitness: We will use a quick sort"""
        """This quick sort will sort the fitness array from lowest to highest.
            Additionally, at the same time it will sort its corresponding population """
        # print numOfPop
        Sort.quick_sort_population(population, fitness, 0, numOfPop)


    # ------------------------------------------------------------------------------
    def split_parents(self,mom, dad, numOfFeatures):
        """Split the parents by the number of split points given by the user"""
        """Check that the new child has at lest 3 features """
        child_one_sum = 0  # Initialize variables to 0
        child_two_sum = 0
        twins=0
        """Due to the limited number of features children could potentially be twins"""
        while (child_one_sum < 3) and (child_two_sum < 3) and twins !=1 :
            """Reset children values"""
            child_one_sum = 0
            child_two_sum = 0
            splitpoint = random.randint(0, numOfFeatures)
            '''found that we did not need to seed the random because we did get
               different values every time we printed it'''
            child_one = concatenate((mom[0:splitpoint], dad[splitpoint:]))
            child_two = concatenate((dad[0:splitpoint], mom[splitpoint:]))
            child_one_sum = child_one.sum()
            child_two_sum = child_two.sum()


        """Return the two children"""
        return child_one, child_two
    def mutation(self,child,numOfFeatures):

        mutation_made = 0
        while mutation_made != 1:
            mutation = random.randint(0,numOfFeatures)
            """Check if the random postion is 0, if it is change it to one"""
            """Else look for another spot"""
            if child[mutation] == 0:
                child[mutation] = 1
                mutation_made=1
        return child

    def Create_GA_Population(self, numOfPop, numOfFea, previousPop, ga_pop, fitness):
        """get the previous population"""
        """Sort population by fitness using a quick sort """
        self.sort_population(previousPop, fitness, numOfPop)
        """ Genetic algorithm population size"""
        ga_population = ga_pop

        new_pop = zeros((ga_population, numOfFea))
        j = 2
        """Get the best two parents add to new pop and make their children and add to new population"""
        new_pop[0] = previousPop[0]
        new_pop[1] = previousPop[1]
        """Make children with parents and add to new population"""
        #new_pop[2], new_pop[3] = self.split_parents(previousPop[0], previousPop[1], numOfFea)

        """split the best for loop to create children best of 10 add to new population"""
        """from 2 till n, it will """
        #counter = 4;
        """Create Children with parents, take parents in 2's, (0,1),(2,3)...(x-1,x)"""
        for x in range(0, ga_population-2, 2):
            new_pop[j], new_pop[j + 1] = self.split_parents(previousPop[x], previousPop[x + 1], numOfFea)
            new_pop[j] = self.mutation(new_pop[j],numOfFea)
            new_pop[j+1] = self.mutation(new_pop[j+1],numOfFea)

            #print str(x) + " " + str(j)
            j = j+2
        """Create new random samples calling the function Create_A_Population """
        #print new_pop[19];

        random_population = self.Create_A_Population(numOfPop - ga_population, numOfFea)

        if ga_population >= 2:
            #print "population greater than 2"
            new_pop = concatenate((new_pop, random_population))
        else:
            new_pop = random_population
            return new_pop

        """return the new population set"""
        return new_pop
        """new_pop = self.Create_A_Population(ga_population, numOfFea)
        random_population = self.Create_A_Population(numOfPop - ga_population, numOfFea)"""

        #return concatenate((new_pop, random_population))
