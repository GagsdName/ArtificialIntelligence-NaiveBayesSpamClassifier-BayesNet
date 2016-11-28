'''
Created on 21-Nov-2016
'''
from os import listdir
from os.path import splitext, join
import string
import pickle
import operator

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
        spamDocs = [ spamDoc for spamDoc in listdir(spamDir) if splitext(spamDoc)[0].startswith('0')]
        for doc in spamDocs:
            self.totalDocs += 1
            self.spamDocs += 1
            print("{}-{}".format('Current document', doc))
            with open(join(spamDir,doc), 'r') as currentDoc:
                # featurePresent is used to mark the presence of features in current document
                featurePresent = set()
                for line in currentDoc:
                    for token in line.lower().translate(replace_punctuation).split():
                        if token not in featurePresent:
                            featurePresent.add(token)
                            featureSpamDocCount = (self.featureDocCount[token][0]+1) if token in self.featureDocCount else 1
                            featureNonSpamDocCount = (self.featureDocCount[token][1]) if token in self.featureDocCount else 0
                            self.featureDocCount[token] = (featureSpamDocCount, featureNonSpamDocCount)
                        if token in stop_tokens or token.isdigit() or len(token)<4: continue
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
        nonspamDocs = [ nonspamDoc for nonspamDoc in listdir(nonSpamDir) if splitext(nonspamDoc)[0].startswith('0')]
        for doc in nonspamDocs:
            self.totalDocs += 1
            self.nonSpamDocs += 1
            print("{}-{}".format('Current document', doc))
            with open(join(nonSpamDir,doc), 'r') as currentDoc:
                # featurePresent is used to mark the presence of features in current document
                featurePresent = set()
                for line in currentDoc:
                    for token in line.lower().translate(replace_punctuation).split():
                        if token not in featurePresent:
                            featurePresent.add(token)
                            featureSpamDocCount = (self.featureDocCount[token][0]) if token in self.featureDocCount else 0
                            featureNonSpamDocCount = (self.featureDocCount[token][1]+1) if token in self.featureDocCount else 1
                            self.featureDocCount[token] = (featureSpamDocCount, featureNonSpamDocCount)
                        if token in stop_tokens or token.isdigit() or len(token)<4: continue
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
        spamTokenFrequency = {}
        nonSpamTokenFrequency = {}
        for token in self.features:
            spamTokenFrequency[token] = self.features[token][0]
            nonSpamTokenFrequency[token] = self.features[token][1]
        mostLikelySpam = sorted(spamTokenFrequency.items(), key=operator.itemgetter(1), reverse=True)
        leastLikelySpam = sorted(nonSpamTokenFrequency.items(), key=operator.itemgetter(1), reverse=True)
        print('Keywords most likely associated with Spam: ')
        for i in range(0, 10):
            print(mostLikelySpam[i][0])
        print('Keywords most likely associated with Non-Spam: ')
        for i in range(0, 10):
            print(leastLikelySpam[i][0])

# Test Training                    
'''
trainer = Trainer()
trainer.trainSpamDocs('./train/spam')
trainer.trainNonSpamDocs('./train/notspam')
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
                'delivered', 'class', 'home', 'right', 'left', 'list', 'colspan']
with open('stop_words.pkl', 'wb') as stop_words_file:
    pickle.dump(set(get_stop_words('en')).union(list(string.punctuation)).union(html_tokens), stop_words_file)
    stop_words_file.close()
'''