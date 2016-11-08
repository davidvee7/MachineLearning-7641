__author__ = 'davidvinegar'

from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import decomposition
from time import time
from sklearn.decomposition import FastICA
from scipy.spatial.distance import cdist, pdist
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import mixture

incomeDatatFilePath = "/Users/davidvinegar/Desktop/Entertainment/GeorgiaTech/MachineLearning/assignment3/sklearnTransform/income_training.csv"
housingDatatFilePath = "/Users/davidvinegar/Desktop/Entertainment/GeorgiaTech/MachineLearning/assignment3/housing_training_70.csv"

#Loads in Data, accepting a string for either the "income" filetype or the "housing" filetype
def initData(fileType):
    if fileType == "income":
        fname = open(incomeDatatFilePath)
    else:
        fname = open(housingDatatFilePath)

    dataset = np.loadtxt(fname, delimiter = ",").astype(np.int64)
    return dataset

#print out statistics for a k means estimator
def bench_k_means(estimator, name, data, fileType):
    t0 = time()
    estimator.fit(data)
    dataset = initData(fileType)
    labels = dataset[:,-1]

    print('% 9s,  %.2fs,   %i,   %.3f,   %.3f,   %.3f,   %.3f,   %.3f,   %.3f,'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=dataset.shape[0])))

# run statistics from start to stop, moving by step.  datatype
# is a string that refers to either "income" or "housing".
def printKMeansResults(start, stop, dataType, clusterCount, step = 1 ):

    dataset = initData(dataType)

    print(79 * '_')

    print('% 9s' % 'init'
          '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

    bench_k_means(KMeans(n_clusters=clusterCount),
                  "k-means ", dataset, dataType)

    for i in range(start,stop,step):
        bench_k_means(KMeans(n_clusters=i),
                      name ="k-means" + str(i), data=dataset, fileType=dataType)

#chart the variance of the data that can be explained by K
#data is a dataset as an ndarray
#kStart and kStop denote the start and stop parameters for the x axis of the k means variance graph
def explainKMeansVariance(data,kStart, kStop, dataSetType = "Housing"):
    k_range = range(kStart,kStop)
    k_means_var = [KMeans(n_clusters=k).fit(data) for k in k_range]

    centroids = [X.cluster_centers_ for X in k_means_var]

    k_euclid = [cdist(data, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke,axis=1) for ke in k_euclid]

    wcss = [sum(d**2) for d in dist]

    tss = sum(pdist(data)**2)/data.shape[0]

    bss = tss-wcss

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_range, bss/tss*100, 'b*-')
    ax.set_ylim((0,100))
    plt.grid(True)
    plt.xlabel('n_clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title(dataSetType + ': Variance Explained vs. k')

    plt.show()

#Run the expectation maximization algorithm, setting its number of components
def doEM(n_components,dataType):
    incomeData = initData(dataType)
    X = incomeData[:,:-1]
    t0 = time()
    em = mixture.GaussianMixture(n_components=n_components)
    em.fit(X)
    predictions = em.predict(X)


    print('% s  %.2fs   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (n_components, (time() - t0),
             metrics.homogeneity_score(incomeData[:,-1], predictions),
             metrics.completeness_score(incomeData[:,-1], predictions),
             metrics.v_measure_score(incomeData[:,-1], predictions),
             metrics.adjusted_rand_score(incomeData[:,-1], predictions),
             metrics.adjusted_mutual_info_score(incomeData[:,-1],  predictions),
             metrics.silhouette_score(X, predictions,
                                      )))

#Run the Principal Componnets Analysis algorithm, providing the string datatype as either "income" or "housing"
def doPCA(dataType):
    np.random.seed(5)

    np.set_printoptions(threshold='nan')

    dataset = initData(dataType)
    X = dataset[:,:-1]

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)

    # print "------------Explained Variance ----------------------"
    # print pca.explained_variance_
    # print "------------Explained Variance Ratio ----------------------"
    #
    # print pca.explained_variance_ratio_

    scale = StandardScaler()
    X = scale.fit_transform(dataset[:,:-1])
    X= pca.fit_transform(X)
    reconstruction = scale.inverse_transform(pca.inverse_transform(X))
    R_Error = sum(map(np.linalg.norm, reconstruction-dataset[:,:-1]))

    # print "------------R_Error----------------------"

    print R_Error

#
def doICA(dataset,num_components):
    np.random.seed(5)
    np.set_printoptions(threshold='nan')

    dataset = initData(dataset)
    X = dataset[:,:-1]

    ica = FastICA(n_components=num_components)


    X = ica.fit_transform(X)

    # print X
    reconstruction = ica.inverse_transform(X)

    # print stats.kurtosis(X)

    # print ica.mixing_
    reconstruction_error = sum(map(np.linalg.norm, reconstruction-dataset[:,:-1]))

    print num_components, reconstruction_error

def doRandomizedProjection(datasetName,n):
    np.random.seed(5)
    np.set_printoptions(threshold='nan')

    dataset = initData(datasetName)
    X = dataset[:,:-1]

    scale = StandardScaler()

    transformer = random_projection.GaussianRandomProjection(n_components=n)

    scaledX = scale.fit_transform(X)
    scaledX = transformer.fit_transform(scaledX)
    # print"printing scaledX"
    # print scaledX
    # print "------***********_----------"
    inverse = np.linalg.pinv(transformer.components_)
    reconstruction = scale.inverse_transform(np.dot(scaledX, inverse.T))

    reconstruction_error = sum(map(np.linalg.norm, reconstruction-dataset[:,:-1]))

    print reconstruction_error

def doUnivariateFeatureSelection(datasetName):
    dataset = initData(datasetName)
    X = dataset[:,:-1]

    y = dataset[:,-1]

    selectKBest = SelectKBest(chi2,k=2).fit(X,y)
    print selectKBest.scores_

#Create the dimension reduction algorithms and return them
def dimReduce(n):
    pca = PCA(n_components=n)
    fastICA = FastICA(n_components=n)
    randomProjection = random_projection.GaussianRandomProjection(n_components=n)
    kBest = SelectKBest(chi2,k=n)

    return pca, fastICA, randomProjection,kBest

#Create clusters using EM and KM using dimension reduced data for a given dataset(either "income" or "housing"
def clusterOnDimenReduced(datasetName):
    dataset = initData(datasetName)
    X = dataset[:,:-1]

    y = dataset[:,-1]
    t0 = time()
    # transformed = PCA(n_components=3).fit_transform(X,y)
    # transformed = FastICA(n_components=3).fit_transform(X,y)

    scale = StandardScaler()

    transformer = random_projection.GaussianRandomProjection(n_components=3)
    scaledX = scale.fit_transform(X)
    # transformed = transformer.fit_transform(scaledX)    #

    transformed = SelectKBest(chi2,k=3).fit_transform(X,y)


    # k_means = KMeans(n_clusters=6)
    # k_means.fit(transformed)
    # bench_k_means(k_means,"K-means ICA",transformed, datasetName)

    n_components = 3
    em = mixture.GaussianMixture(n_components=3)
    em.fit(transformed)

    predictions = em.predict(transformed)


    print('% s  %.2fs   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (n_components, (time() - t0),
             metrics.homogeneity_score(dataset[:,-1], predictions),
             metrics.completeness_score(dataset[:,-1], predictions),
             metrics.v_measure_score(dataset[:,-1], predictions),
             metrics.adjusted_rand_score(dataset[:,-1], predictions),
             metrics.adjusted_mutual_info_score(dataset[:,-1],  predictions),
             metrics.silhouette_score(X, predictions,
                                      )))

def basicNN(datasetName):
    dataset = initData(datasetName)
    X = dataset[:,:-1]

    y = dataset[:,-1]
    t0 = time()
    clf = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)

    clf.fit(X,y)
    t1 = time()

    predictions= clf.predict(X)
    timed = t1-t0
    print "----Accuracy of neural network trained with backprop----"
    print "Trained in " + str(metrics.accuracy_score(y,predictions)) + " seconds"
    print timed

def runNNonDimReduced(datasetType):

    if datasetType == "income":
        n = 3
    else:
        n = 14
    dimReduced = dimReduce(n)

    dataset = initData(datasetType)
    X = dataset[:,:-1]

    y = dataset[:,-1]

    clfPCA = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clfICA = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clfRP = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clfFS = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)

    pcaData = dimReduced[0].fit_transform(X,y)
    icaData = dimReduced[1].fit_transform(X,y)
    randomProjectionData = dimReduced[2].fit_transform(X,y)
    featureSelectedData = dimReduced[3].fit_transform(X,y)
    timeInit = time()
    clfPCA.fit(pcaData,y)
    pcaTime =time()- timeInit

    timeInit=time()
    clfICA.fit(pcaData,y)
    icaTime = time() - timeInit

    timeInit = time()
    clfRP.fit(pcaData,y)
    rpTime = time() - timeInit

    timeInit=time()
    clfFS.fit(pcaData,y)
    flsTime = time()-timeInit


    #Run against training set
    predictionsPCA= clfPCA.predict(X)
    predictionsICA= clfICA.predict(X)
    predictionsRP= clfRP.predict(X)
    predictionsFS= clfFS.predict(X)

    print "----Training Set Accuracy----"
    print "PCA " + str(metrics.accuracy_score(y,predictionsPCA))
    print "ICA " + str(metrics.accuracy_score(y,predictionsICA))
    print "RP " + str(metrics.accuracy_score(y,predictionsRP))
    print "FS " + str(metrics.accuracy_score(y,predictionsFS))

    print "----Training Time---------"
    print "PCA " + str(pcaTime)
    print "ICA " + str(icaTime)
    print "RP " + str(rpTime)
    print "FS " + str(flsTime)

    incomeTestDatatFilePath = "/Users/davidvinegar/Desktop/Entertainment/GeorgiaTech/MachineLearning/assignment3/sklearnTransform/income_test.csv"

    fname = open(incomeTestDatatFilePath)
    testDataset = np.loadtxt(fname, delimiter = ",").astype(np.int64)

    testX = testDataset[:,:-1]
    testY = testDataset[:,-1]
    testPredictionsPCA= clfPCA.predict(testX)
    testPredictionsICA= clfICA.predict(testX)
    testPredictionsRP= clfRP.predict(testX)
    testPredictionsFS= clfFS.predict(testX)

    #Run against Test Set
    print "----Test Set Accuracy----"
    print "PCA " + str(metrics.accuracy_score(testY,testPredictionsPCA))
    print "ICA " + str(metrics.accuracy_score(testY,testPredictionsICA))
    print "RP " + str(metrics.accuracy_score(testY,testPredictionsRP))
    print "FS " + str(metrics.accuracy_score(testY,testPredictionsFS))


def clusterTrainedNN():

    dataset = initData("income")

    X = dataset[:,:-1]
    y = dataset[:,-1]

    kmeansSix = KMeans(n_clusters=6)
    kmeansSeven = KMeans(n_clusters=7)
    kmeansFive = KMeans(n_clusters=5)

    em5 = mixture.GaussianMixture(n_components=5)

    fiveTransformed = kmeansFive.fit_transform(X,y)
    sixTransformed = kmeansSix.fit_transform(X,y)
    sevenTransformed = kmeansSeven.fit_transform(X,y)

    # em5Tranformed = em5.fit_transform(X,y)

    clfSeven = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clfSix = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clfFive = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)

    clfEm5 = MLPClassifier(solver='lbfgs',alpha = 1e-5,hidden_layer_sizes=(5,2), random_state=1)

    clfSeven.fit(sevenTransformed,y)
    clfSix.fit(sixTransformed,y)

    init = time()
    clfFive.fit(fiveTransformed,y)
    trainingTimeclfFive = time()- init

    init2 = time()
    clfEm5.fit(X,y)
    trainingTimeEM = time() - init2
        #Run against training set
    predictionsSeven= clfSeven.predict(sevenTransformed)
    predictionsSix= clfSix.predict(sixTransformed)
    predictionsFive= clfFive.predict(fiveTransformed)

    # predictionsEm5 = clfEm5.predict(em5Tranformed)

    print "----Training Set Accuracy----"
    print "KM " + str(metrics.accuracy_score(y,predictionsFive))
    print "EM " + str(metrics.accuracy_score(y,predictionsSix))

    print "-------training times--------"
    print "KM " + str(trainingTimeclfFive)
    print "EM " + str(trainingTimeEM)
    # print metrics.accuracy_score(y,predictionsEm5)

    incometestDatatFilePath = "/Users/davidvinegar/Desktop/Entertainment/GeorgiaTech/MachineLearning/assignment3/sklearnTransform/income_test.csv"

    fname = open(incometestDatatFilePath)
    testDataset = np.loadtxt(fname, delimiter = ",").astype(np.int64)

    testX = testDataset[:,:-1]
    testY = testDataset[:,-1]

    clusteredTestFive = kmeansFive.fit_transform(testX,testY)
    clusteredTestSix = kmeansSix.fit_transform(testX,testY)
    clusteredTestSeven = kmeansSeven.fit_transform(testX,testY)

    em5.fit(testX,testY)

    testPredictionsFive= clfFive.predict(clusteredTestFive)
    testPredictionsSix= clfSix.predict(clusteredTestSix)
    testPredictionsSeven= clfSeven.predict(clusteredTestSeven)

    testPredictionEm5 = clfEm5.predict(testX)
    #Run against Test Set
    print "----Test Set Accuracy----"
    print "KM " + str(metrics.accuracy_score(testY,testPredictionsSix))
    print "EM " + str(metrics.accuracy_score(testY, testPredictionEm5))

# getHousingStatistics for kmeans

print "********  KMeans Housing   ************"
printKMeansResults(2,6,"housing",5,1)
print "********  KMeans Income   ************"
printKMeansResults(2,6,"income",5,1)

print ""
print "********  EM Housing   ************"
print('% 9s' % 'init' '  time   homo   compl  v-meas     ARI AMI  silhouette')
for i in range(2,10,2):
    doEM(i,"housing")
print "********  EM Income   ************"
for i in range(2,10,2):
    doEM(i,"income")

print ""
print "********  PCA Housing (Reconstruction Error)  ************"
doPCA("housing")
print "********  PCA Income  (Reconstruction Error)  ************"
doPCA("income")

print ""
print "********  ICA Income  (Reconstruction Error) ************"
for i in range(1,4):
    doICA("income",i)
print "********  ICA Housing (Reconstruction Error)   ************"
for i in range(1,14):
    doICA("housing",i)

print ""
print "********  RP Housing (Reconstruction Error)  ************"
doRandomizedProjection("housing",13)
print "********  RP Income (Reconstruction Error)  ************"
doRandomizedProjection("income",3)

print ""
print "********  FS Housing (Feature Ranking)  ************"
doUnivariateFeatureSelection("housing")
print "********  FS Income (Feature Ranking)  ************"
doUnivariateFeatureSelection("income")

print "********  Cluster on Dimensionality Reduced Data Housing ************"
print('% 9s' % 'init' '  time   homo   compl  v-meas     ARI AMI  silhouette')
clusterOnDimenReduced("housing")

print "********  Cluster on Dimensionality Reduced Data Income ************"
print('% 9s' % 'init' '  time   homo   compl  v-meas     ARI AMI  silhouette')
clusterOnDimenReduced("income")

basicNN("income")

print "********  Neural Network Trained with Dimensionality Reduced Income Data************"

runNNonDimReduced("income")

print "********  Neural Network Trained with clusteredIncome Data************"
clusterTrainedNN()