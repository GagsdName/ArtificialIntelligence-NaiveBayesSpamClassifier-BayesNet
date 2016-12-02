'''
Created on 21-Nov-2016
'''
from __future__ import division
from os import listdir
from os.path import splitext, join, isfile
from math import log
from node import Node
import string
import pickle
import operator
from sys import setrecursionlimit
setrecursionlimit(10000)

class Trainer:
    def __init__(self):
        self.totalDocs = 0
        self.spamDocs = 0
        self.nonSpamDocs = 0
        self.totalTokens = 0
        self.spamTokens = 0
        self.nonSpamTokens = 0
        self.features = {}
        self.featureDocCount = {}
        self.mostLikelySpamKeywords = []
    
    def get_stop_words(self):
        with open('stop_words.pkl', 'rb') as stop_words_file:
            return pickle.load(stop_words_file)
# Pre-processing steps:
# 1) Tokenization 2) Stopwords, punctuation removal 3) Lower case conversion
    def trainSpamDocs(self, spamDir):
        print('Training on Spam Emails inside /train/spam ...')
        print('Extracting Stop-Words...')
        stop_tokens = self.get_stop_words()
        replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        # Consider only those files that begin with '0' -> Avoid noise introduction from cmds file
        spamDocs = [ spamDoc for spamDoc in listdir(spamDir) if splitext(spamDoc)[0].isdigit()]
        for doc in spamDocs:
            self.totalDocs += 1
            self.spamDocs += 1
            print("{}-{}".format('Current document', doc))
            with open(join(spamDir,doc), 'r') as currentDoc:
                # featurePresent is used to mark the presence of features in current document
                featurePresent = set()
                for line in currentDoc:
                    for token in line.lower().translate(replace_punctuation).split():
                        if token in stop_tokens or token[0].isdigit() or len(token)<4 or len(token)>10: continue
                        if token not in featurePresent:
                            featurePresent.add(token)
                            featureSpamDocCount = (self.featureDocCount[token][0]+1) if token in self.featureDocCount else 1
                            featureNonSpamDocCount = (self.featureDocCount[token][1]) if token in self.featureDocCount else 0
                            self.featureDocCount[token] = (featureSpamDocCount, featureNonSpamDocCount)
                        self.totalTokens += 1
                        self.spamTokens += 1
                        spamCount = (self.features[token][0]+1) if token in self.features else 1
                        nonSpamCount = (self.features[token][1]) if token in self.features else 0
                        self.features[token] = (spamCount, nonSpamCount)
                currentDoc.close()
        print("{}-{}".format('Total Spam Docs', self.spamDocs))
        print('Training on Spam Emails completed!')
        # print(self.features)
    
    def trainNonSpamDocs(self, nonSpamDir):
        print('Training on Non-Spam Emails inside /train/notspam ...')
        stop_tokens = self.get_stop_words()
        replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        # Consider only those files that begin with '0' -> Avoid noise introduction from cmds file
        nonspamDocs = [ nonspamDoc for nonspamDoc in listdir(nonSpamDir) if splitext(nonspamDoc)[0].isdigit()]
        for doc in nonspamDocs:
            self.totalDocs += 1
            self.nonSpamDocs += 1
            print("{}-{}".format('Current document', doc))
            with open(join(nonSpamDir,doc), 'r') as currentDoc:
                # featurePresent is used to mark the presence of features in current document
                featurePresent = set()
                for line in currentDoc:
                    for token in line.lower().translate(replace_punctuation).split():
                        if token in stop_tokens or token[0].isdigit() or len(token)<4 or len(token)>10: continue
                        if token not in featurePresent:
                            featurePresent.add(token)
                            featureSpamDocCount = (self.featureDocCount[token][0]) if token in self.featureDocCount else 0
                            featureNonSpamDocCount = (self.featureDocCount[token][1]+1) if token in self.featureDocCount else 1
                            self.featureDocCount[token] = (featureSpamDocCount, featureNonSpamDocCount)
                        self.totalTokens += 1
                        self.nonSpamTokens += 1
                        nonSpamCount = (self.features[token][1]+1) if token in self.features else 1
                        spamCount = (self.features[token][0]) if token in self.features else 0
                        self.features[token] = (spamCount, nonSpamCount)
                currentDoc.close()
        print("{}-{}".format('Total Non-Spam Docs', self.nonSpamDocs))    
        print('Training on Non-Spam Emails completed!')
        # print(self.features)
        
    def generateModelFile(self, outputFile):
        with open(outputFile+'.pkl', 'wb') as modelFile:
            pickle.dump(self, modelFile)
            modelFile.close()
            
    def findLikelySpamKeywords(self):
        print(len(self.features))
        spamTokenFrequency = {}
        nonSpamTokenFrequency = {}
        for token in self.features:
            spamTokenFrequency[token] = self.features[token][0]
            nonSpamTokenFrequency[token] = self.features[token][1]
        mostLikelySpam = sorted(spamTokenFrequency.items(), key=operator.itemgetter(1), reverse=True)
        self.mostLikelySpamKeywords = mostLikelySpam
        leastLikelySpam = sorted(nonSpamTokenFrequency.items(), key=operator.itemgetter(1), reverse=True)
        print('Keywords most likely associated with Spam: ')
        for i in range(0, 10):
            print(mostLikelySpam[i][0])
        print('Keywords most likely associated with Non-Spam: ')
        for i in range(0, 10):
            if leastLikelySpam[i] in self.mostLikelySpamKeywords:
                del leastLikelySpam[i]
            print(leastLikelySpam[i][0])
        
####################################################################################################
    # This method generates the DT recursively
    def DT_Induction_Binary(self, topFeatures, targetDir, trainDocs, trainLabels):
        if len(topFeatures)<44000:
            return Node('Spam') if self.features[topFeatures[0][0]][0]>self.features[topFeatures[0][0]][1] else Node('Not-Spam')
                
        # If all are Spam emails
        if 0 not in trainLabels:
            return Node('Spam')
        elif 1 not in trainLabels:
            return Node('Not-Spam')
        root = Node(topFeatures.pop(0)[0])
        spamPredicates = []
        spamLabels = []
        nonSpamPredicates = []
        nonSpamLabels = []
        # Check whether the current feature is associated with Spam or Non-Spam
        spamProb = self.features[root.value][0]/self.spamTokens
        nonSpamProb = self.features[root.value][1]/self.nonSpamTokens
        featureLabel = True if spamProb >= nonSpamProb else False
        for i in range(0, len(trainDocs)):
            if isfile(join(targetDir+'/spam/', trainDocs[i])):
                with open(join(targetDir+'/spam/', trainDocs[i])) as temp:
                    if root.value in temp.read():
                        if featureLabel:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(1)
                        else:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(1)
                    else:
                        if featureLabel:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(1)
                        else:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(1)
                    temp.close()
            else:
                with open(join(targetDir+'/notspam/', trainDocs[i])) as temp:
                    if root.value in temp.read():
                        if featureLabel:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(0)
                        else:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(0)
                    else:
                        if featureLabel:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(0)
                        else:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(0)
                    temp.close()
        root.left = self.DT_Induction_Binary(topFeatures, targetDir, spamPredicates, spamLabels)
        root.right = self.DT_Induction_Binary(topFeatures, targetDir, nonSpamPredicates, nonSpamLabels)
        return root   
####################################################################################################

####################################################################################################
    # This method generates the DT recursively
    def DT_Induction_Continuous(self, topFeatures, targetDir, trainDocs, trainLabels):
        if len(topFeatures)<44000:
            return Node('Spam') if self.features[topFeatures[0][0]][0]>self.features[topFeatures[0][0]][1] else Node('Not-Spam')
        # If all are Spam emails
        if 0 not in trainLabels:
            return Node('Spam')
        elif 1 not in trainLabels:
            return Node('Not-Spam')
        root = Node(topFeatures.pop(0)[0])
        spamPredicates = []
        spamLabels = []
        nonSpamPredicates = []
        nonSpamLabels = []
        # Check whether the current feature is associated with Spam or Non-Spam
        spamProb = self.features[root.value][0]/self.spamTokens
        nonSpamProb = self.features[root.value][1]/self.nonSpamTokens
        featureLabel = True if spamProb >= nonSpamProb else False
        for i in range(0, len(trainDocs)):
            if isfile(join(targetDir+'/spam/', trainDocs[i])):
                with open(join(targetDir+'/spam/', trainDocs[i])) as temp:
                    if root.value in temp.read():
                        if featureLabel:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(1)
                        else:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(1)
                    else:
                        if featureLabel:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(1)
                        else:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(1)
                    temp.close()
            else:
                with open(join(targetDir+'/notspam/', trainDocs[i])) as temp:
                    if root.value in temp.read():
                        if featureLabel:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(0)
                        else:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(0)
                    else:
                        if featureLabel:
                            nonSpamPredicates.append(trainDocs[i])
                            nonSpamLabels.append(0)
                        else:
                            spamPredicates.append(trainDocs[i])
                            spamLabels.append(0)
                    temp.close()
        root.left = self.DT_Induction_Continuous(topFeatures, targetDir, spamPredicates, spamLabels)
        root.right = self.DT_Induction_Continuous(topFeatures, targetDir, nonSpamPredicates, nonSpamLabels)
        return root   
####################################################################################################

####################################################################################################
# This function prints the node of a DT
    def printTree(self, node, level):
        if level<1 or node is None:
            return
        print('( {} {} {} )'.format(node.value, node.left.value if node.left is not None else None, node.right.value if node.right is not None else None))
        if node.left is not None:
            self.printTree(node.left, level-1)
        if node.right is not None:
            self.printTree(node.right, level-1)
####################################################################################################

####################################################################################################
# This method is used to generate model file for the DT and dump it into a pickle
    def generateDTModelFile(self, outputFile, root):
        with open(outputFile+'.pkl', 'wb') as modelFile:
            pickle.dump(root, modelFile)
            modelFile.close()
####################################################################################################

####################################################################################################
# This method builds a Decision Tree based on the Binary Features
# Each feature is considered to be an attribute and then for each attribute, we calculate the average
# disorder. Each attribute has a PRESENT or NOT_PRESENT value, so each node in the DT will have two
# branches coming out. If an attribute is PRESENT, mark the document as SPAM otherwise move to the
# left node of the tree.
# HOW TO INTERPRET THE DT: Each node of the tree is printed as a 3-valued tuple in the main console.
# The first value represents the value of that node, 2nd value represents the left branch and the
# 3rd value represents the right branch.
    def buildDTBinary(self, trainDir, outputFile):
        # Calculate the entropy score for each feature
        print('Generating the Binary Decision Tree...')
        disorderMap = {}
        for feature in self.features:
            presentCount = self.featureDocCount[feature][0]+self.featureDocCount[feature][1]
            notPresentCount = self.totalDocs - presentCount
            disorderScore = 0.0
            # First calculate for IF TOKEN PRESENT
            temp = 0
            for j in range(0,2):
                if presentCount>0 and self.featureDocCount[feature][j]>0:
                    temp += (-1)*(self.featureDocCount[feature][j]/presentCount)*log(self.featureDocCount[feature][j]/presentCount)
            temp *= presentCount/(presentCount+notPresentCount)
            disorderScore += temp
            # Now calculate for IF TOKEN NOT PRESENT
            temp = 0
            for j in range(0,2):
                if notPresentCount>0 and self.featureDocCount[feature][j]>0:
                    temp += (-1)*(((self.spamDocs if j==0 else self.nonSpamDocs) - self.featureDocCount[feature][j])/notPresentCount)*log(((self.spamDocs if j==0 else self.nonSpamDocs) - self.featureDocCount[feature][j])/notPresentCount)
            temp *= notPresentCount/(presentCount+notPresentCount)
            disorderScore += temp
            disorderMap[feature] = disorderScore
            
        # Now sort the disorderMap in increasing order of disorderScore
        topFeatures = sorted(disorderMap.items(), key=operator.itemgetter(1))
        print(len(topFeatures))
        # Generate a list of all training documents to be used for building DT
        trainDocs = [ spamDoc for spamDoc in listdir(trainDir+'/spam/') if splitext(spamDoc)[0].startswith('0')]
        spamLabels = [1] * len(trainDocs)
        trainDocs += [ nonSpamDoc for nonSpamDoc in listdir(trainDir+'/notspam/') if splitext(nonSpamDoc)[0].startswith('0')]
        nonSpamLabels = [0] * (len(trainDocs)-len(spamLabels))
        trainLabels = spamLabels + nonSpamLabels
        # Create a Decision Tree
        root = self.DT_Induction_Binary(topFeatures, trainDir, trainDocs, trainLabels)
        self.printTree(root, 5)
        print('Saving the DT model file into pickle...')
        self.generateDTModelFile(outputFile, root)
        print('Generating the Binary Decision Tree complete...')
####################################################################################################

####################################################################################################
# This method builds a Decision Tree based on the Continuous Features
# HOW TO INTERPRET THE DT: Each node of the tree is printed as a 3-valued tuple in the main console.
# The first value represents the value of that node, 2nd value represents the left branch and the
# 3rd value represents the right branch.
    def buildDTContinuous(self, trainDir, outputFile):
        # Calculate the entropy score for each feature
        print('Generating the Continuous Decision Tree...')
        disorderMap = {}
        for feature in self.features:
            disorderScore = 0.0
            # First calculate for IF TOKEN PRESENT
            temp = 0
            if self.features[feature][0]>0 and self.spamTokens>0:
                temp += (-1)*(self.features[feature][0]/self.spamTokens)*log(self.features[feature][0]/self.spamTokens)
            temp *= self.spamTokens/self.totalTokens
            disorderScore += temp
            # Now calculate for IF TOKEN NOT PRESENT
            temp = 0
            if self.features[feature][1]>0 and self.nonSpamTokens>0:
                temp += (-1)*(self.features[feature][1]/self.nonSpamTokens)*log(self.features[feature][1]/self.nonSpamTokens)
            temp *= self.nonSpamTokens/self.totalTokens
            disorderScore += temp
            disorderMap[feature] = disorderScore
            
        # Now sort the disorderMap in increasing order of disorderScore
        topFeatures = sorted(disorderMap.items(), key=operator.itemgetter(1))
        print(len(topFeatures))
        # Generate a list of all training documents to be used for building DT
        trainDocs = [ spamDoc for spamDoc in listdir(trainDir+'/spam/') if splitext(spamDoc)[0].startswith('0')]
        spamLabels = [1] * len(trainDocs)
        trainDocs += [ nonSpamDoc for nonSpamDoc in listdir(trainDir+'/notspam/') if splitext(nonSpamDoc)[0].startswith('0')]
        nonSpamLabels = [0] * (len(trainDocs)-len(spamLabels))
        trainLabels = spamLabels + nonSpamLabels
        # Create a Decision Tree
        root = self.DT_Induction_Continuous(topFeatures, trainDir, trainDocs, trainLabels)
        self.printTree(root, 5)
        print('Saving the DT model file into pickle...')
        self.generateDTModelFile(outputFile, root)
        print('Generating the Continuous Decision Tree complete...')
####################################################################################################


# Test Training                    
'''
trainer = Trainer()
trainer.trainSpamDocs('./train/spam')
trainer.trainNonSpamDocs('./train/notspam')
print(trainer.features)
trainer.findLikelySpamKeywords()
print(trainer.featureDocCount['path'])
print(trainer.featureDocCount['dogma'])
print(trainer.featureDocCount['single'])
print(trainer.featureDocCount['transfer'])
print(trainer.featureDocCount['drop'])
'''
# Below is a list containing common HTML keywords responsible to introduce
# noise in the training. To filter out such tokens from being added into the
# dictionary during training, we have dumped this list into the same
# stop_words .pkl pickle file and load it during training.
##### Creating a Pickle file to be used in training #####
'''
html_tokens = ['font', 'size', 'http', 'width', 'nbsp', 'color', 'height', 'href',
                'face', 'align', 'localhost', 'arial', 'table', 'content', 'fork',
                'received', 'list', 'style', 'center', 'border', 'type', 'html',
                'text', 'esmtp', 'name', 'xent', 'verdana', 'click', 'bgcolor', 'cnet',
                'cellspacing', 'cellpadding', 'version', 'zdnet', 'clickthru', 'admin',
                'online', 'span', 'subject', 'helvetica', 'mail', 'option', 'value',
                'zzzz', 'email', 'mailto', 'images', 'body', 'ffffff', 'date', 'sans',
                'serif', 'will', 'message', 'postfix', 'header', 'example', 'charset',
                'input', 'return', 'smtp', 'valign', 'halign', 'listinfo', 'netnoteinc',
                'mime', 'subscribe', 'unsubscribe', 'request', 'help', 'delivery',
                'delivered', 'class', 'home', 'right', 'left', 'list', 'colspan', 'path',
                'sender', 'receiver', 'recipient', 'title', 'rssfeeds', 'tests']
with open('stop_words.pkl', 'wb') as stop_words_file:
    pickle.dump(set(get_stop_words('en')).union(list(string.punctuation)).union(html_tokens), stop_words_file)
    stop_words_file.close()
'''