import time                  # provides timing for benchmarks
from numpy   import *        # provides complex math and array functions
#from sklearn import svm     # provides Support Vector Regression
import csv
import math
import sys
import hashlib

import FromDataFileMLR
import mlr


#------------------------------------------------------------------------------


def r2(y, yHat):
    #Coefficient of determination
    """ y is the target and yHat is the prediction, regression analysis"""
    """ calculate the total sum of squares, this is proportional to the variance of the data"""
    numer = ((y - yHat)**2).sum()       # Residual Sum of Squares
    """Calculate the residual sum of squares"""
    denom = ((y - y.mean())**2).sum()  # denominator = 16.81256
    """coeficient of determination is R^2 = 1- (rs/ts)"""
    r2 = 1 - numer/denom
    #print str(numer)+" "+str(denom)+ " "+str(numer/denom)+" "+str(r2)+" "+str(y)+" "+str(yHat)+" "+str(y.mean())
    return r2
#------------------------------------------------------------------------------

def r2Pred(yTrain, yTest, yHatTest):
    """prediction of y for the training set"""
    """ from the prediction set, subtract target y value from the yHattest"""
    numer = ((yHatTest - yTest)**2).sum()
    """subtraction the y mean value from the training set"""
    denom = ((yTest - yTrain.mean())**2).sum()
    """"""
    r2Pred = 1 - numer/denom
    return r2Pred

#------------------------------------------------------------------------------

def cv_predict(model, set_x, set_y):
    # Predict using cross validation.
    """Prediction of the model, an MLR objet"""
    yhat = empty_like(set_y)
    """for the number of rows"""
    for idx in range(0, yhat.shape[0]):
        """deleting row around the 0 axis"""
        #print set_x
        train_x = delete(set_x, idx, axis=0)
        train_y = delete(set_y, idx, axis=0)
        modelName = model.fit(train_x, train_y)
        yhat[idx] = model.predict(set_x[idx])
    return yhat

#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013

def calc_fitness(xi, Y, Yhat, c=2):
    """
    Calculate fitness of a prediction.
    xi : array_like -- Mask of features to measure fitness of. Must be of dtype bool.
    c : float       -- Adjustment parameter.
    """

    p = sum(xi)   # Number of selected parameters (features)
    n = len(Y)    # Sample size, Y of training + validation data

    numer = ((Y - Yhat)**2).sum()/n   # Mean square error, always greater than zero
    pcn = p*(c/n)  # Num of features * 2/ number of rows (training + validation)(30)
    if pcn >= 1:   # if the result of pcn is greater than 1 then is invalid
        return 1000
    denom = (1 - pcn)**2
    """calculate fitness number/(1 - sum(xi/n))"""
    theFitness = numer/denom
    return theFitness

#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def InitializeTracks():
    """Set all arrays to empty and return them"""
    trackDesc = {}
    trackFitness = {}
    trackModel = {}
    trackR2 = {}
    trackQ2 = {}
    trackR2PredValidation = {}
    trackR2PredTest = {}

    return  trackDesc, trackFitness, trackModel, trackR2, trackQ2, \
            trackR2PredValidation, trackR2PredTest
#------------------------------------------------------------------------------
#Ahmad Hadaegh: Modified  on: July 16, 2013
def initializeYDimension():
    """Initializing to empty all target data/y array"""
    yTrain = {}
    yHatTrain = {}
    yHatCV = {}
    yValidation = {}
    yHatValidation = {}
    yTest = {}
    yHatTest = {}
    return yTrain, yHatTrain, yHatCV, yValidation, yHatValidation, yTest, yHatTest 
#------------------------------------------------------------------------------
def OnlySelectTheOnesColumns(popI):

    numOfFea = 385#popI.shape[0]  # get total number of features
    xi = zeros(numOfFea)  # create an array with the total number of features
    for j in range(numOfFea):
       xi[j] = popI[j]  # copying elements from one array to another

    xi = xi.nonzero()[0]  # select the features that are not Zero
    xi = xi.tolist()  # create a list/ array of features and return it
    return xi
 
#------------------------------------------------------------------------------

def validate_model(model, fileW, population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    
    numOfPop = population.shape[0] # get the population based on the number of features selected
    """Create an array based on the population size"""
    fitness = zeros(numOfPop)
    c = 2
    """ initialize booleans for false=0 and true =1"""
    false = 0
    true = 1
    predictive = false

    """Initialize all arrays/matrices, """
    trackDesc, trackFitness,trackModel,trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest  = InitializeTracks()

    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest = initializeYDimension()

    unfit = 1000
    itFits = 1
    for i in range(numOfPop):
        """Get columns that have a value of one and eliminate the rest"""
        xi = OnlySelectTheOnesColumns(population[i])
        """Store data in a hash table for fast look up and encrypt the data using sha1"""
        idx = hashlib.sha1(array(xi)).digest()

        X_train_masked = TrainX.T[xi].T

        X_validation_masked = ValidateX.T[xi].T
        X_test_masked = TestX.T[xi].T
      
        try:
            model_desc = model.fit(X_train_masked, TrainY)
        except:
            return unfit, fitness
        
        # Computed predicted values
        Yhat_cv = cv_predict(model, X_train_masked, TrainY)    # Cross Validation
        Yhat_validation = model.predict(X_validation_masked)
        Yhat_test = model.predict(X_test_masked)
            
        # Compute R2 statistics (Prediction for Valiation and Test set)
        q2_loo = r2(TrainY, Yhat_cv)
        q2_loo = FromDataFileMLR.getTwoDecPoint(q2_loo)

        r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
        r2pred_validation = FromDataFileMLR.getTwoDecPoint(r2pred_validation)

        r2pred_test = r2Pred(TrainY, TestY, Yhat_test)
        r2pred_test = FromDataFileMLR.getTwoDecPoint(r2pred_test)
                      
        Y_fitness = append(TrainY, ValidateY)
        Yhat_fitness = append(Yhat_cv, Yhat_validation)
            
        fitness[i] = calc_fitness(xi, Y_fitness, Yhat_fitness, c)

        if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
            # if it's not worth recording, just return the fitness
            print "ending the program because of predictive is: ", predictive
            continue
            
        # Compute predicted Y_hat for training set.
        Yhat_train = model.predict(X_train_masked)
        r2_train = r2(TrainY, Yhat_train)

        idxLength = len(xi)

        # store stats
        trackDesc[idx] = str(xi)

        trackFitness[idx] = FromDataFileMLR.getTwoDecPoint(fitness[i])

        trackModel[idx] = model_desc
        
        trackR2[idx] = FromDataFileMLR.getTwoDecPoint(r2_train)
        trackQ2[idx] = FromDataFileMLR.getTwoDecPoint(q2_loo)
        trackR2PredValidation[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_validation)
        trackR2PredTest[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_test)

        yTrain[idx] = TrainY.tolist()

        yHatTrain[idx] = Yhat_train.tolist()
        for i in range(len(yHatTrain[idx])):
            yHatTrain[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTrain[idx][i])

        yHatCV[idx] = Yhat_cv.tolist()
        for i in range(len(yHatCV[idx])):
            yHatCV[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatCV[idx][i])

        yValidation[idx] = ValidateY.tolist()

        yHatValidation[idx] = Yhat_validation.tolist()
        for i in range(len(yHatValidation[idx])):
            yHatValidation[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatValidation[idx][i])

        yTest[idx] = TestY.tolist()

        yHatTest[idx] = Yhat_test.tolist()
        for i in range(len(yHatTest[idx])):
            yHatTest[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTest[idx][i])

    write(model,fileW, trackDesc, trackFitness, trackModel, trackR2,\
                trackQ2,trackR2PredValidation, trackR2PredTest)

    return itFits, fitness
#------------------------------------------------------------------------------  

def validate_single_model(model, vector, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    #numOfPop = population.shape[0]  # get the population based on the number of features selected
    """Create an array based on the population size"""
    #fitness = zeros(numOfPop)
    fitness =0
    c = 2
    """ initialize booleans for false=0 and true =1"""
    false = 0
    true = 1
    predictive = false

    """Initialize all arrays/matrices, """
    trackDesc, trackFitness, trackModel, trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest = InitializeTracks()

    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest = initializeYDimension()

    unfit = 1000
    itFits = 1

    """Get columns that have a value of one and eliminate the rest"""
    xi = OnlySelectTheOnesColumns(vector)
    """Store data in a hash table for fast look up and encrypt the data using sha1"""
    idx = hashlib.sha1(array(xi)).digest()

    X_train_masked = TrainX.T[xi].T

    X_validation_masked = ValidateX.T[xi].T
    X_test_masked = TestX.T[xi].T

    try:
        model_desc = model.fit(X_train_masked, TrainY)
    except:
        return unfit, fitness

    # Computed predicted values
    Yhat_cv = cv_predict(model, X_train_masked, TrainY)  # Cross Validation
    Yhat_validation = model.predict(X_validation_masked)
    Yhat_test = model.predict(X_test_masked)

    # Compute R2 statistics (Prediction for Valiation and Test set)
    q2_loo = r2(TrainY, Yhat_cv)
    q2_loo = FromDataFileMLR.getTwoDecPoint(q2_loo)

    r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
    r2pred_validation = FromDataFileMLR.getTwoDecPoint(r2pred_validation)

    r2pred_test = r2Pred(TrainY, TestY, Yhat_test)
    r2pred_test = FromDataFileMLR.getTwoDecPoint(r2pred_test)

    Y_fitness = append(TrainY, ValidateY)
    Yhat_fitness = append(Yhat_cv, Yhat_validation)

    fitness = calc_fitness(xi, Y_fitness, Yhat_fitness, c)

    if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
        # if it's not worth recording, just return the fitness
        print "ending the program because of predictive is: ", predictive

    # Compute predicted Y_hat for training set.
    Yhat_train = model.predict(X_train_masked)
    r2_train = r2(TrainY, Yhat_train)

    idxLength = len(xi)

    # store stats
    trackDesc[idx] = str(xi)

    trackFitness[idx] = FromDataFileMLR.getTwoDecPoint(fitness)

    trackModel[idx] = model_desc

    trackR2[idx] = FromDataFileMLR.getTwoDecPoint(r2_train)
    trackQ2[idx] = FromDataFileMLR.getTwoDecPoint(q2_loo)
    trackR2PredValidation[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_validation)
    trackR2PredTest[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_test)

    yTrain[idx] = TrainY.tolist()

    yHatTrain[idx] = Yhat_train.tolist()
    for i in range(len(yHatTrain[idx])):
        yHatTrain[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTrain[idx][i])

    yHatCV[idx] = Yhat_cv.tolist()
    for i in range(len(yHatCV[idx])):
        yHatCV[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatCV[idx][i])

    yValidation[idx] = ValidateY.tolist()

    yHatValidation[idx] = Yhat_validation.tolist()
    for i in range(len(yHatValidation[idx])):
        yHatValidation[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatValidation[idx][i])

    yTest[idx] = TestY.tolist()

    yHatTest[idx] = Yhat_test.tolist()
    for i in range(len(yHatTest[idx])):
        yHatTest[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTest[idx][i])

    return fitness

def write(model,fileW, trackDesc, trackFitness, trackModel, trackR2,\
                trackQ2,trackR2PredValidation, trackR2PredTest):
    
    for key in trackFitness.keys():
        fileW.writerow([trackDesc[key], trackFitness[key], trackModel[key], \
            trackR2[key], trackQ2[key], trackR2PredValidation[key], trackR2PredTest[key]])


    #fileOut.close()

#------------------------------------------------------------------------------
def append_to_file(fileW, trackDesc, trackFitness, trackModel, trackR2,\
                trackQ2,trackR2PredValidation, trackR2PredTest):
    """Append new vector to file"""
    for key in trackFitness.keys():
        fileW.writerow([trackDesc[key], trackFitness[key], trackModel[key], \
                    trackR2[key], trackQ2[key], trackR2PredValidation[key], trackR2PredTest[key]])

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
    fileOut = file(file_name, 'ab')
    fileW = csv.writer(fileOut)

    fileW.writerow(['Descriptor ID', 'Fitness', 'Model','R2', 'Q2', \
            'R2Pred_Validation', 'R2Pred_Test'])

    return fileW
#---------------------------------------------------------------------------
def validate_model_and_append(model, fileW, vector, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    # numOfPop = population.shape[0]  # get the population based on the number of features selected
    """Create an array based on the population size"""
    # fitness = zeros(numOfPop)
    fitness = 0
    c = 2
    """ initialize booleans for false=0 and true =1"""
    false = 0
    true = 1
    predictive = false

    """Initialize all arrays/matrices, """
    trackDesc, trackFitness, trackModel, trackR2, trackQ2, \
    trackR2PredValidation, trackR2PredTest = InitializeTracks()

    yTrain, yHatTrain, yHatCV, yValidation, \
    yHatValidation, yTest, yHatTest = initializeYDimension()

    unfit = 1000
    itFits = 1

    """Get columns that have a value of one and eliminate the rest"""
    xi = OnlySelectTheOnesColumns(vector)
    """Store data in a hash table for fast look up and encrypt the data using sha1"""
    idx = hashlib.sha1(array(xi)).digest()

    X_train_masked = TrainX.T[xi].T

    X_validation_masked = ValidateX.T[xi].T
    X_test_masked = TestX.T[xi].T

    try:
        model_desc = model.fit(X_train_masked, TrainY)
    except:
        return unfit, fitness

    # Computed predicted values
    Yhat_cv = cv_predict(model, X_train_masked, TrainY)  # Cross Validation
    Yhat_validation = model.predict(X_validation_masked)
    Yhat_test = model.predict(X_test_masked)

    # Compute R2 statistics (Prediction for Valiation and Test set)
    q2_loo = r2(TrainY, Yhat_cv)
    q2_loo = FromDataFileMLR.getTwoDecPoint(q2_loo)

    r2pred_validation = r2Pred(TrainY, ValidateY, Yhat_validation)
    r2pred_validation = FromDataFileMLR.getTwoDecPoint(r2pred_validation)

    r2pred_test = r2Pred(TrainY, TestY, Yhat_test)
    r2pred_test = FromDataFileMLR.getTwoDecPoint(r2pred_test)

    Y_fitness = append(TrainY, ValidateY)
    Yhat_fitness = append(Yhat_cv, Yhat_validation)

    fitness = calc_fitness(xi, Y_fitness, Yhat_fitness, c)

    if predictive and ((q2_loo < 0.5) or (r2pred_validation < 0.5) or (r2pred_test < 0.5)):
        # if it's not worth recording, just return the fitness
        print "ending the program because of predictive is: ", predictive

    # Compute predicted Y_hat for training set.
    Yhat_train = model.predict(X_train_masked)
    r2_train = r2(TrainY, Yhat_train)

    idxLength = len(xi)

    # store stats
    trackDesc[idx] = str(xi)

    trackFitness[idx] = FromDataFileMLR.getTwoDecPoint(fitness)

    trackModel[idx] = model_desc

    trackR2[idx] = FromDataFileMLR.getTwoDecPoint(r2_train)
    trackQ2[idx] = FromDataFileMLR.getTwoDecPoint(q2_loo)
    trackR2PredValidation[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_validation)
    trackR2PredTest[idx] = FromDataFileMLR.getTwoDecPoint(r2pred_test)

    yTrain[idx] = TrainY.tolist()

    yHatTrain[idx] = Yhat_train.tolist()
    for i in range(len(yHatTrain[idx])):
        yHatTrain[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTrain[idx][i])

    yHatCV[idx] = Yhat_cv.tolist()
    for i in range(len(yHatCV[idx])):
        yHatCV[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatCV[idx][i])

    yValidation[idx] = ValidateY.tolist()

    yHatValidation[idx] = Yhat_validation.tolist()
    for i in range(len(yHatValidation[idx])):
        yHatValidation[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatValidation[idx][i])

    yTest[idx] = TestY.tolist()

    yHatTest[idx] = Yhat_test.tolist()
    for i in range(len(yHatTest[idx])):
        yHatTest[idx][i] = FromDataFileMLR.getTwoDecPoint(yHatTest[idx][i])

    write(model, fileW, trackDesc, trackFitness, trackModel, trackR2, \
          trackQ2, trackR2PredValidation, trackR2PredTest)

