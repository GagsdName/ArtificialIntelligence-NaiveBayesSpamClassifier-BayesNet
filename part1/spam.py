'''
Created on 21-Nov-2016
'''
# ./spam mode technique dataset-directory model-file

from __future__ import division
import sys
import pickle
import string
import operator
from os.path import isfile ,splitext, join
from os import listdir
from trainer import Trainer
from math import log
from node import Node


####################################################################################################
# This method calculates the Prior Probabilities P(Spam) and P(Non-Spam) from Training Data
def getPriors(trainingData):
    spamProb = trainingData.spamDocs / trainingData.totalDocs
    nonSpamProb = trainingData.nonSpamDocs / trainingData.totalDocs
    return (spamProb, nonSpamProb)
####################################################################################################

####################################################################################################
# This method calculates the accuracy of a classifier.
# Input: Benchmark array and Predictions array
# Output: Accuracy value
def findAccuracy(benchmark, predictions):
    correctCount = 0
    for i in range(0, len(benchmark)):
        if benchmark[i] == predictions[i]:
            correctCount += 1
    return (correctCount/len(benchmark))   
####################################################################################################

####################################################################################################
# This method prints the Confusion Matrix
# Input: A list of correct/benchmark labels for the test set and a corresponding list of
# predicted labels
def printConfusionMatrix(benchmark, predictions):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for i in range(0, len(benchmark)):
        if benchmark[i]==1:
            if predictions[i]==1:
                truePos += 1
            else:
                falseNeg += 1
        else:
            if predictions[i]==0:
                trueNeg += 1
            else:
                falsePos += 1
    print('----- CONFUSION MATRIX -----')
    print('------- SPAM | NOTSPAM -----')
    print("{}{}{}{}".format('SPAM    ', truePos, '     ', falseNeg))
    print("{}{}{}{}".format('NOTSPAM ', falsePos, '     ', trueNeg))
####################################################################################################

####################################################################################################
# This method calculates the accuracy based on Naive Bayes Binary Classifier model
# This model only considers whether a token in test document occurred in training data or not.
# If the token did not occur at all, it is ignored completely. If it occurs in the SPAM email, then
# the spam count is incremented, if it occurs in non-spam email, non-spam count is incremented.
# After doing this for all the tokens in test document, we compare the spam and non-spam count and
# whichever is greater, we label the document accordingly.
# Accuracy achieved with this basic model: 94%
# After removing the stopwords and punctuations, the accuracy drops to 85%
# After proper tokenization and removing the HTML keywords during training, the accuracy improved
# to 96.8%. This kind of accuracy is not possible with a Binary Bayes Classifier, so there is 
# something we are doing wrong here.
####################################################################################################
def naive_bayes_binary(targetDir, benchmark):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    predictions = []
    print("{}-{}".format('Total Test Docs', len(benchmark)))
    # Running Bayes Classifier with Binary features
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./spam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './spam/', doc), 'r') as currentDoc:
            spamCount = 0
            nonSpamCount = 0
            for line in currentDoc:
                for token in line.lower().translate(replace_punctuation).split():
                    if token in trainingData.features:
                        if trainingData.features[token][0]>0:
                            spamCount += 1
                        if trainingData.features[token][1]>0:
                            nonSpamCount += 1
            currentDoc.close()
        if spamCount >= nonSpamCount:
            # Mark this document as Spam
            predictions.append(1)
        else:
            predictions.append(0)
        
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./notspam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            spamCount = 0
            nonSpamCount = 0
            for line in currentDoc:
                for token in line.lower().translate(replace_punctuation).split():
                    if token in trainingData.features:
                        if trainingData.features[token][0]>0:
                            spamCount += 1
                        if trainingData.features[token][1]>0:
                            nonSpamCount += 1
            currentDoc.close()
        if spamCount >= nonSpamCount:
            # Mark this document as Spam
            predictions.append(1)
        else:
            predictions.append(0)
            
    # print("{}-{}".format('Length of Benchmark', len(benchmark)))
    # print("{}-{}".format('Length of Predictions', len(predictions)))  
    # print("{}-{}".format('Correct Predictions', correctCount))      
    print("{}-{}".format('Naive Bayes Binary Classifier Accuracy', findAccuracy(benchmark, predictions)))
    printConfusionMatrix(benchmark, predictions)
####################################################################################################

####################################################################################################
# This method calculates the accuracy based on Naive Bayes Continuous Classifier model
# Ignore the unseen tokens from test document
# P(Spam/w1,w2,..,wn) = P(Spam).P(w1/Spam).P(w2/Spam)...P(wn/Spam)
# P(Non-Spam/w1,w2,..,wn) = P(Non-Spam).P(w1/Non-Spam).P(w2/Non-Spam)...P(wn/Non-Spam)
# Mark the test document as Spam if P(Spam/w1,w2,..,wn)>=P(Non-Spam/w1,w2,..,wn); Non-Spam otherwise
# Accuracy achieved with this basic model: 63%
# After removing the stopwords and punctuations, the accuracy drops to 43%
def naive_bayes_prob(trainingData, targetDir, benchmark):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    predictions = []
    # Running Bayes Classifier with Binary features
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./spam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './spam/', doc), 'r') as currentDoc:
            spamProb, nonSpamProb = getPriors(trainingData)
            for line in currentDoc:
                for token in line.lower().translate(replace_punctuation).split():
                    # Ignore the tokens not seen in training data
                    if token in trainingData.features:
                        if trainingData.features[token][0]>0:
                            spamProb *= trainingData.features[token][0] / trainingData.spamTokens
                        if trainingData.features[token][1]>0:
                            nonSpamProb *= trainingData.features[token][1] / trainingData.nonSpamTokens
            currentDoc.close()
        if spamProb >= nonSpamProb:
            # Mark this document as Spam
            predictions.append(1)
        else:
            predictions.append(0)
        
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./notspam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            spamProb = trainingData.spamDocs / trainingData.totalDocs
            nonSpamProb = trainingData.nonSpamDocs / trainingData.totalDocs
            for line in currentDoc:
                for token in line.lower().translate(replace_punctuation).split():
                    if token in trainingData.features:
                        if trainingData.features[token][0]>0:
                            spamProb *= trainingData.features[token][0] / trainingData.spamTokens
                        if trainingData.features[token][1]>0:
                            nonSpamProb *= trainingData.features[token][1] / trainingData.nonSpamTokens
            currentDoc.close()
        if spamProb >= nonSpamProb:
            # Mark this document as Spam
            predictions.append(1)
        else:
            predictions.append(0)
            
    # Calculate accuracy
    correctCount = 0
    for i in range(0, len(benchmark)):
        if benchmark[i] == predictions[i]:
            correctCount += 1
    # print("{}-{}".format('Length of Benchmark', len(benchmark)))
    # print("{}-{}".format('Length of Predictions', len(predictions)))  
    # print("{}-{}".format('Correct Predictions', correctCount))      
    print("{}-{}".format('Naive Bayes Continuous Classifier Accuracy', findAccuracy(benchmark, predictions)))
    printConfusionMatrix(benchmark, predictions)
####################################################################################################

####################################################################################################
# This method builds a Decision Tree based on the Binary Features
# Each feature is considered to be an attribute and then for each attribute, we calculate the average
# disorder. Each attribute has a PRESENT or NOT_PRESENT value, so each node in the DT will have two
# branches coming out. If an attribute is PRESENT, mark the document as SPAM otherwise move to the
# left node of the tree.
def buildDTBinary(trainingData, targetDir, benchmark):
    # Calculate the entropy score for each feature
    disorderMap = {}
    for feature in trainingData.features:
        presentCount = trainingData.featureDocCount[feature][0]+trainingData.featureDocCount[feature][1]
        notPresentCount = trainingData.totalDocs - presentCount
        disorderScore = 0.0
        # First calculate for IF TOKEN PRESENT
        temp = 0
        for j in range(0,2):
            if presentCount>0 and trainingData.featureDocCount[feature][j]>0:
                temp += (-1)*(trainingData.featureDocCount[feature][j]/presentCount)*log(trainingData.featureDocCount[feature][j]/presentCount)
        temp *= presentCount/(presentCount+notPresentCount)
        disorderScore += temp
        # Now calculate for IF TOKEN NOT PRESENT
        temp = 0
        for j in range(0,2):
            if notPresentCount>0 and trainingData.featureDocCount[feature][j]>0:
                temp += (-1)*(((trainingData.spamDocs if j==0 else trainingData.nonSpamDocs) - trainingData.featureDocCount[feature][j])/notPresentCount)*log(((trainingData.spamDocs if j==0 else trainingData.nonSpamDocs) - trainingData.featureDocCount[feature][j])/notPresentCount)
        temp *= notPresentCount/(presentCount+notPresentCount)
        disorderScore += temp
        disorderMap[feature] = disorderScore
        
    # Now sort the disorderMap in increasing order of disorderScore
    topFeatures = sorted(disorderMap.items(), key=operator.itemgetter(1))
    # print(topFeatures)
    # Create a Decision Tree
    root = Node(topFeatures[0][0])
    currentNode = root
    for i in range(1, 5):
        temp = Node(topFeatures[i][0])
        currentNode.right = Node('Spam')
        currentNode.left = temp
        currentNode = temp
    currentNode.right = Node('Spam')
    currentNode.left = Node('Not-Spam')
    
    currentNode = root
    for i in range(0, 4):
        currentNode.printNode()
        currentNode = currentNode.left
    return root
####################################################################################################

####################################################################################################
# This method runs the Decision Tree Binary classifier on Test set and prints the results
# Accuracy received with the DT level 10: 33%
def runDTBinary(dt, targetDir, benchmark):
    predictions = []
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./spam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './spam/', doc), 'r') as currentDoc:
            tokenSet = set()
            for line in currentDoc:
                tokenSet.union(line.lower().translate(replace_punctuation).split())
            currentNode = dt
            flag = False
            while currentNode.value != 'Not-Spam':
                if currentNode.value in tokenSet:
                    flag = True
                    break
                else:
                    currentNode = currentNode.left
            predictions.append(1 if flag else 0)
            currentDoc.close()
            
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./notspam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            flag = False
            while currentNode.value != 'Not-Spam':
                if currentNode.value in emailText:
                    flag = True
                    break
                else:
                    currentNode = currentNode.left
            predictions.append(1 if flag else 0)
            currentDoc.close()
    print("{}-{}".format('Decision Tree Binary Classifier Accuracy', findAccuracy(benchmark, predictions)))
    printConfusionMatrix(benchmark, predictions)
####################################################################################################

if len(sys.argv) != 5:
    print('Invalid number of input arguments.')
    print('./spam mode technique dataset-directory model-file')
    exit()
mode = sys.argv[1]
technique = sys.argv[2]
datasetDir = './' + sys.argv[3]
modelFile = './' + sys.argv[4]

if mode == 'train':
    trainData = Trainer()
    trainData.trainSpamDocs(datasetDir+'/spam/')
    trainData.trainNonSpamDocs(datasetDir+'/notspam/')
    trainData.generateModelFile(modelFile)
    trainData.findLikelySpamKeywords()
    exit()
    
if mode == 'test':
    # Load the model-file
    if not isfile(modelFile+'.pkl'):
        print('Model File does not exist. Please run test mode first.')
        exit()
    with open(modelFile+'.pkl', 'rb') as trainingDataFile:
        trainingData = pickle.load(trainingDataFile)
    spamProb = trainingData.spamDocs/trainingData.totalDocs
    nonSpamProb = trainingData.nonSpamDocs/trainingData.totalDocs
    # print(spamProb)
    # print(nonSpamProb)
    # Collect the TEST documents from spam and notspam folders
    # Mark Spam Documents with '1' and Non-Spam Documents with '0'
    testDocs = [ doc for doc in listdir(datasetDir+'./spam/') if splitext(doc)[0].startswith('0')]
    spam = [1] * len(testDocs)
    testDocs += [ doc for doc in listdir(datasetDir+'./notspam/') if splitext(doc)[0].startswith('0')]
    nonspam = [0] * (len(testDocs)-len(spam))
    benchmark = spam + nonspam
    
    # Run the Naive-Bayes Binary Classifier
    # naive_bayes_binary(datasetDir, benchmark)
    # naive_bayes_prob(trainingData, datasetDir, benchmark)
    runDTBinary(buildDTBinary(trainingData, datasetDir, benchmark), datasetDir, benchmark)