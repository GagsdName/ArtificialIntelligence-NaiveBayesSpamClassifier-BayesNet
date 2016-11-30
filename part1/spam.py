'''
Created on 21-Nov-2016
'''
# ./spam mode technique dataset-directory model-file

from __future__ import division
import sys
import pickle
import string
from os.path import isfile ,splitext, join
from os import listdir
from trainer import Trainer

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
# This method runs the Decision Tree Binary classifier on Test set and prints the results
# Accuracy received with the DT level 10: 33%. The decision tree is heavily biased towards Not-Spam.
# Apparently there were some bugs in the way we generated the DT above. After correcting them,
# the accuracy we get is 65%. The total features in our dictionary are around 47k. To generate the
# DT, we have only used the top 2.5k features. Using more features hit the Python recursion limit
# and sometimes Python crashed while saving the DT into a pickle.
def runDTBinary(dt, targetDir, benchmark):
    predictions = []
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./spam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './spam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            while currentNode.value is not None:
                if currentNode.value in emailText:
                    if currentNode.left.value == 'Spam':
                        predictions.append(1)
                        break
                    elif currentNode.left.value == 'Not-Spam':
                        predictions.append(0)
                        break
                    else:
                        currentNode = currentNode.left
                        continue
                else:
                    if currentNode.right.value == 'Spam':
                        predictions.append(1)
                        break
                    elif currentNode.left.value == 'Not-Spam':
                        predictions.append(0)
                        break
                    else:
                        currentNode = currentNode.right
        currentDoc.close()
            
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'./notspam/') if splitext(doc)[0].startswith('0')]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            while currentNode.value is not None:
                if currentNode.value in emailText:
                    if currentNode.left.value == 'Spam':
                        predictions.append(1)
                        break
                    elif currentNode.left.value == 'Not-Spam':
                        predictions.append(0)
                        break
                    else:
                        currentNode = currentNode.left
                        continue
                else:
                    if currentNode.right.value == 'Spam':
                        predictions.append(1)
                        break
                    elif currentNode.left.value == 'Not-Spam':
                        predictions.append(0)
                        break
                    else:
                        currentNode = currentNode.right
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

if mode == 'train' and technique == 'bayes':
    trainData = Trainer()
    trainData.trainSpamDocs(datasetDir+'/spam/')
    trainData.trainNonSpamDocs(datasetDir+'/notspam/')
    trainData.generateModelFile(modelFile)
    trainData.findLikelySpamKeywords()
    exit()
    
if mode == 'train' and technique == 'dt':
    trainData = Trainer()
    trainData.trainSpamDocs(datasetDir+'/spam/')
    trainData.trainNonSpamDocs(datasetDir+'/notspam/')
    trainData.buildDTBinary(datasetDir, modelFile)
    exit()
    
if mode == 'test':
    # Load the model-file
    if not isfile(modelFile+'.pkl'):
        print('Model File does not exist. Please run test mode first.')
        exit()
    with open(modelFile+'.pkl', 'rb') as trainingDataFile:
        trainingData = pickle.load(trainingDataFile)
    # spamProb = trainingData.spamDocs/trainingData.totalDocs
    # nonSpamProb = trainingData.nonSpamDocs/trainingData.totalDocs
    # print(spamProb)
    # print(nonSpamProb)
    # Collect the TEST documents from spam and notspam folders
    # Mark Spam Documents with '1' and Non-Spam Documents with '0'
    testDocs = [ doc for doc in listdir(datasetDir+'./spam/') if splitext(doc)[0].startswith('0')]
    spam = [1] * len(testDocs)
    testDocs += [ doc for doc in listdir(datasetDir+'./notspam/') if splitext(doc)[0].startswith('0')]
    nonspam = [0] * (len(testDocs)-len(spam))
    benchmark = spam + nonspam
    
    if technique == 'bayes':
        # Run the Naive-Bayes Binary Classifiers
        naive_bayes_binary(datasetDir, benchmark)
        naive_bayes_prob(trainingData, datasetDir, benchmark)
    elif technique == 'dt':
        runDTBinary(trainingData, datasetDir, benchmark)
    else:
        print('Invalid Technique. Accepted values: \'bayes\' or \'dt\'')
        exit()
    # Run the Decision Tree Classifiers
