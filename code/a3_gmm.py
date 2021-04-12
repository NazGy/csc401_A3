from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''

    # print(x.shape, myTheta.Sigma[m].shape)

    bm = np.sum(0.5 * (x ** 2) * (myTheta.Sigma[m] ** -1) - myTheta.mu[m] * x * (myTheta.Sigma[m] ** -1), axis=1)

    bm += preComputedForM[m]

    return -bm

    
def log_p_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    if len(preComputedForM) == 0:
        d = len(myTheta.mu[m])
        preComputedForM = np.sum((myTheta.mu ** 2) / (2 * myTheta.Sigma), axis=1) + d/2 * np.log(2 * np.pi) + 0.5 * np.log(np.prod(myTheta.Sigma, axis=1))

    # print(np.log(myTheta.omega[m]))

    numer = np.log(myTheta.omega[m]) + log_b_m_x( m, x, myTheta, preComputedForM=preComputedForM)
    
    # denom = np.array([myTheta.omega[k] * np.exp(log_b_m_x(k, x, myTheta, preComputedForM=preComputedForM)) for k in range(myTheta.omega.shape[0])])

    log_Bs = np.array([log_b_m_x(k, x, myTheta, preComputedForM=preComputedForM) for k in range(myTheta.omega.shape[0])])

    denom = logsumexp(a=log_Bs, b=myTheta.omega, axis=0)

    # print(numer.shape, denom.shape)
    # print(denom)
    # denom = np.sum(denom, axis=0)

    pmx = numer - denom

    return pmx

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    # px = 0

    # preComputedForM = np.sum((myTheta.mu ** 2) / (2 * (myTheta.Sigma))) + d/2 * np.log(2 * np.pi) + 0.5 * np.log(np.prod(myTheta.Sigma))

    # prod = np.log(myTheta.omega) * log_Bs

    # print("logBs", log_Bs.shape)

    px = logsumexp(a=log_Bs, b=myTheta.omega, axis=0)

    lik = np.sum(px)
    return lik

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, Sigma)'''
    myTheta = theta( speaker, M, X.shape[1] )
    # Initialize myTheta with random values
    d = X.shape[1]
    T = X.shape[0]

    myTheta.mu = X[np.random.randint(0, T - 1, size=M)]
    myTheta.Sigma += 1
    myTheta.omega = np.ones((M, 1))/M


    
    

    # print("preM", preComputedForM.shape)

    # eq1 = np.zeros((M, T))
    # eq2 = np.zeros((M, T))
    # for m in range(M):
    #     eq1[m] = log_b_m_x(m, X, myTheta, preComputedForM)
    #     eq2[m] = log_p_m_x(m, X, myTheta, preComputedForM)

    # print(eq1)

    i = 0
    prev_L = - float('inf')
    improvement = float('inf')
    while i < maxIter and improvement >= epsilon:
        preComputedForM = np.sum((myTheta.mu ** 2) / (2 * myTheta.Sigma), axis=1) + d/2 * np.log(2 * np.pi) + 0.5 * np.log(np.prod(myTheta.Sigma, axis=1))
        eq1 = np.zeros((M, T))
        eq2 = np.zeros((M, T))
        for m in range(M):
            eq1[m] = log_b_m_x(m, X, myTheta, preComputedForM)
            eq2[m] = log_p_m_x(m, X, myTheta, preComputedForM)
        # print("eq1", eq1.shape, "eq2", eq2.shape)
        # print("u", myTheta.mu.shape, "sig", myTheta.Sigma.shape, "w", myTheta.omega.shape)
        L = logLik(eq1, myTheta)

        print("L:", L)

        pmx = np.exp(eq2) #(M, t)

        sumOverT = np.sum(pmx, axis=1) 

        # print(pmx)

        myTheta.omega = np.expand_dims(sumOverT, axis=1)/T #(M, 1)

        myTheta.mu = ((pmx @ X).T / sumOverT).T #(M, d)

        # print(myTheta.mu ** 2)
        # print(((pmx @ (X ** 2)).T / sumOverT).T)

        myTheta.Sigma = ((pmx @ (X ** 2)).T / sumOverT).T - myTheta.mu ** 2

        improvement = L - prev_L
        prev_L = L
        i = i + 1

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    vals = []
    d = mfcc.shape[1]
    T = mfcc.shape[0]

    for i in range(len(models)):
        model = models[i]
        M = model.mu.shape[0]
        eq1 = np.zeros((M, T))
        preComputedForM = np.sum((model.mu ** 2) / (2 * model.Sigma), axis=1) + d/2 * np.log(2 * np.pi) + 0.5 * np.log(np.prod(model.Sigma, axis=1))
        
        for m in range(M):
            eq1[m] = log_b_m_x(m, mfcc, model, preComputedForM)

        vals.append((logLik(eq1, model), i, model))

    vals.sort(reverse=True)
    
    print("[", correctID, "]")
    for i in range(k):
        curr = vals[i]
        print("[", curr[2].name,"] ", "[", curr[0], "]")
    
    bestModel = vals[0][1]

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    # arr = np.array([[1, 2], [3, 4]]) ** 2
    # print(arr)

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 40
    epsilon = 0.0
    maxIter = 10
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

    print(accuracy)

