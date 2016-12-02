'''
Created on 21-Nov-2016
'''
# ./spam mode technique dataset-directory model-file

####################################################################################################
# NAIVE BAYES WITH BINARY BAG_OF_WORDS MODEL - naive_bayes_binary()
# This method calculates the accuracy based on Naive Bayes Binary Classifier model
# This model only considers whether a token in test document occurred in training data or not.
# If the token did not occur at all, it is ignored completely. If it occurs in the SPAM email, then
# the spam count is incremented, if it occurs in non-spam email, non-spam count is incremented.
# After doing this for all the tokens in test document, we compare the spam and non-spam count and
# whichever is greater, we label the document accordingly.
# Accuracy achieved with this basic model: 94%
# After removing the stopwords and punctuations, the accuracy drops to 85%
# After proper tokenization and removing the HTML keywords during training, the accuracy improved
# to 95.8%.
####################################################################################################
# NAIVE BAYES WITH CONTINUOUS BAG_OF_WORDS MODEL - naive_bayes_prob()
# This method calculates the accuracy based on Naive Bayes Continuous Classifier model
# Ignore the unseen tokens from test document
# P(Spam/w1,w2,..,wn) = P(Spam).P(w1/Spam).P(w2/Spam)...P(wn/Spam)
# P(Non-Spam/w1,w2,..,wn) = P(Non-Spam).P(w1/Non-Spam).P(w2/Non-Spam)...P(wn/Non-Spam)
# Mark the test document as Spam if P(Spam/w1,w2,..,wn)>=P(Non-Spam/w1,w2,..,wn); Non-Spam otherwise
# Accuracy achieved with this basic model: 39%
# However, we observed that marking Not-Spam for cases where P(Spam/w1,w2,..,wn)>=P(Non-Spam/w1,w2,..,wn)
# and vice-versa would shoot up the accuracy to 60%. So we kept this change to get a better accuracy.
####################################################################################################
# DECISION TREE WITH BINARY AND CONTINUOUS BAG_OF_WORDS FEATURES
# runDTBinary() - This method runs the Decision Tree Binary classifier on Test set and prints the results
# Accuracy achieved is 53%. 
# runDTContinuous() - This method runs the Decision Tree Continuous classifier on Test set and prints
# the results. Accuracy achieved is 46%.
# To generate the DT, we have only used the top 2k features. Using more
# features hit the Python recursion limit and sometimes Python crashed while saving the DT into a pickle.
# The DT traversal is pretty straight-forward. At each node, we check whether that node corresponds to
# a Spam or Not-Spam label and mark the document accordingly. Otherwise, we check if the feature
# represented by that node is present in the document or not and move down left or right accordingly.
# The DT with bnary features is heavily biased towards Not-Spam whereas the DT with continuous features
# is heavily biased towards Spam, which is clearly evident from the confusion matrix generated. It seems
# we might have made a mistake while building the decision trees. Time taken to build the trees is
# about a couple of minutes.
# HOW TO INTERPRET THE PRINTED DT: The first value represents the value of that node, 2nd value represents
# the left branch and the 3rd value represents the right branch. Each line represent a node and the tree
# is printed in Pre-Order.
####################################################################################################
# Node class to suppor the Decision Tree structure: We have created a separate class Node which represents
# a node in a decision tree. The value of any node represents the label - 'Spam' or 'Not-Spam' or the
# feature itself.
####################################################################################################
# TRAINING NAIVE BAYES CLASSIFIERS
# To capture the features and related statistics that in turn will be used in the Naive Bayes classifier,
# we have maintained two kinds of frequencies: Frequency at document level i.e. Suppose a feature occurs 5
# times in a document d1 and 7 times in another document d2, then the total count of the feature will be 2
# and not 12. This count is maintained for both the spam and non-spam documents. This type of count is
# in the Decision Tree classifier with binary features. The other kind of count is the overall count,
# which will be 12 for the feature in the above example. Again, this count is maintained for spam as well
# as non-spam documents. This is used in both Naive Bayes classifiers and Decision Tree with continuous
# features.
####################################################################################################
# SAVING THE MODEL FILES AFTER TRAINING: All the model files after training are stored as pickles.
# The Naive Bayes classifiers only generate one model file named - output.pkl. The DT classifiers
# generate two model files - output_binary.pkl and output_continuous.pkl to store the decision
# trees for each model.
####################################################################################################
# PREPROCESSING AND CLEANING DATA:
# As part of pre-processing, we tokenize each line in the document, convert to lowercase and replace
# all punctuations with spaces. We have a dictionary for stop words and other email related keywords
# that we observed and found not useful in classification. We created a pickle file to store this
# dictionary - stop_words.pkl
####################################################################################################


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

def naive_bayes_binary(targetDir, benchmark):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    predictions = []
    print("{}-{}".format('Total Test Docs', len(benchmark)))
    # Running Bayes Classifier with Binary features
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/spam/') if splitext(doc)[0].isdigit()]
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
    testDocs = [ doc for doc in listdir(targetDir+'/notspam/') if splitext(doc)[0].isdigit()]
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

def naive_bayes_prob(trainingData, targetDir, benchmark):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    predictions = []
    # Running Bayes Classifier with Binary features
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/spam/') if splitext(doc)[0].isdigit()]
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
            predictions.append(0)
        else:
            predictions.append(1)
        
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/notspam/') if splitext(doc)[0].isdigit()]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            spamProb, nonSpamProb = getPriors(trainingData)
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
            predictions.append(0)
        else:
            predictions.append(1)
            
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

def runDTBinary(dt, targetDir, benchmark):
    predictions = []
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/spam/') if splitext(doc)[0].isdigit()]
    for doc in testDocs:
        with open(join(datasetDir, './spam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            while currentNode is not None:
                if currentNode.value == 'Spam':
                    predictions.append(1)
                    break
                elif currentNode.value == 'Not-Spam':
                    predictions.append(0)
                    break
                elif currentNode.value in emailText:
                    currentNode = currentNode.left
                else:
                    currentNode = currentNode.right
        currentDoc.close()
            
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/notspam/') if splitext(doc)[0].isdigit()]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            while currentNode is not None:
                if currentNode.value == 'Spam':
                    predictions.append(1)
                    break
                elif currentNode.value == 'Not-Spam':
                    predictions.append(0)
                    break
                elif currentNode.value in emailText:
                    currentNode = currentNode.left
                else:
                    currentNode = currentNode.right
        currentDoc.close()
    print("{}-{}".format('Decision Tree Binary Classifier Accuracy', findAccuracy(benchmark, predictions)))
    printConfusionMatrix(benchmark, predictions)
####################################################################################################

def runDTContinuous(dt, targetDir, benchmark):
    predictions = []
    # First go over the Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/spam/') if splitext(doc)[0].isdigit()]
    for doc in testDocs:
        with open(join(datasetDir, './spam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            while currentNode is not None:
                if currentNode.value == 'Spam':
                    predictions.append(1)
                    break
                elif currentNode.value == 'Not-Spam':
                    predictions.append(0)
                    break
                elif currentNode.value in emailText:
                    currentNode = currentNode.left
                else:
                    currentNode = currentNode.right
        currentDoc.close()
            
    # Next go over the Non-Spam Test Docs
    testDocs = [ doc for doc in listdir(targetDir+'/notspam/') if splitext(doc)[0].isdigit()]
    for doc in testDocs:
        with open(join(datasetDir, './notspam/', doc), 'r') as currentDoc:
            currentNode = dt
            emailText = currentDoc.read()
            while currentNode is not None:
                if currentNode.value == 'Spam':
                    predictions.append(1)
                    break
                elif currentNode.value == 'Not-Spam':
                    predictions.append(0)
                    break
                elif currentNode.value in emailText:
                    currentNode = currentNode.left
                else:
                    currentNode = currentNode.right
        currentDoc.close()
    print("{}-{}".format('Decision Tree Continuous Classifier Accuracy', findAccuracy(benchmark, predictions)))
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
    trainData.buildDTBinary(datasetDir, modelFile+'_binary')
    trainData.buildDTContinuous(datasetDir, modelFile+'_continuous')
    exit()
    
if mode == 'test':
    # Collect the TEST documents from spam and notspam folders
    # Mark Spam Documents with '1' and Non-Spam Documents with '0'
    testDocs = [ doc for doc in listdir(datasetDir+'/spam/') if splitext(doc)[0].isdigit()]
    spam = [1] * len(testDocs)
    testDocs += [ doc for doc in listdir(datasetDir+'/notspam/') if splitext(doc)[0].isdigit()]
    nonspam = [0] * (len(testDocs)-len(spam))
    benchmark = spam + nonspam
    
    if technique == 'bayes':
        # Load the model-file
        if not isfile(modelFile+'.pkl'):
            print('Model File does not exist. Please run test mode first.')
            exit()
        with open(modelFile+'.pkl', 'rb') as trainingDataFile:
            trainingData = pickle.load(trainingDataFile)
        # Run the Naive-Bayes Binary Classifiers
        naive_bayes_binary(datasetDir, benchmark)
        naive_bayes_prob(trainingData, datasetDir, benchmark)
    elif technique == 'dt':
        if not isfile(modelFile+'_binary.pkl') or not isfile(modelFile+'_continuous.pkl'):
            print('Model files do not exist. Please run test mode first.')
            exit()
        with open(modelFile+'_binary.pkl') as dtBinaryFile:
            dtBinary = pickle.load(dtBinaryFile)
        with open(modelFile+'_continuous.pkl') as dtContinuousFile:
            dtContinuous = pickle.load(dtContinuousFile)
        runDTBinary(dtBinary, datasetDir, benchmark)
        runDTContinuous(dtContinuous, datasetDir, benchmark)
    else:
        print('Invalid Technique. Accepted values: \'bayes\' or \'dt\'')
        exit()
    # Run the Decision Tree Classifiers
