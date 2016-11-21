'''
Created on 21-Nov-2016
'''
from os import listdir
from os.path import splitext, join

class Trainer:
    def __init__(self):
        self.totalTokens = 0
        self.spamTokens = 0
        self.nonSpamTokens = 0
        self.features = {}
        
    def trainSpamDocs(self):
        spamDir = './train/spam/'
        print('Training on Spam Emails inside /train/spam ...')
        # Consider only those files that begin with '0' -> Avoid noise introduction from cmds file
        spamDocs = [ spamDoc for spamDoc in listdir(spamDir) if splitext(spamDoc)[0].startswith('0')]
        for doc in spamDocs:
            print("{}-{}".format('Current document', doc))
            with open(join(spamDir,doc), 'r') as currentDoc:
                for line in currentDoc:
                    for token in line.split():
                        self.totalTokens += 1
                        self.spamTokens += 1
                        spamCount = (self.features[token][0]+1) if token in self.features else 1
                        nonSpamCount = (self.features[token][1]) if token in self.features else 0
                        self.features[token] = (spamCount, nonSpamCount)
                currentDoc.close()
            
        print('Training on Spam Emails completed!')
        # print(self.features)
    
    def trainNonSpamDocs(self):
        nonSpamDir = './train/notspam/'
        print('Training on Non-Spam Emails inside /train/notspam ...')
        # Consider only those files that begin with '0' -> Avoid noise introduction from cmds file
        nonspamDocs = [ nonspamDoc for nonspamDoc in listdir(nonSpamDir) if splitext(nonspamDoc)[0].startswith('0')]
        for doc in nonspamDocs:
            print("{}-{}".format('Current document', doc))
            with open(join(nonSpamDir,doc), 'r') as currentDoc:
                for line in currentDoc:
                    for token in line.split():
                        self.totalTokens += 1
                        self.nonSpamTokens += 1
                        nonSpamCount = (self.features[token][0]+1) if token in self.features else 1
                        spamCount = (self.features[token][1]) if token in self.features else 0
                        self.features[token] = (spamCount, nonSpamCount)
                currentDoc.close()
            
        print('Training on Non-Spam Emails completed!')
        print(self.features)

''' Test Training                    
trainer = Trainer()
trainer.trainSpamDocs()
trainer.trainSpamDocs()
'''