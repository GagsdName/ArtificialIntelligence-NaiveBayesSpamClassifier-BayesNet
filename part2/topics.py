import sys, os, re, pickle,json,heapq, enum
from os import listdir
from os.path import isfile, join, walk
import numpy as np
import math

freq_dict={}
topic_documents = {}
total_topics = 20 #assuming as given in problem statement

Topics = {'atheism':1, 'autos':2, 'baseball':3, 'christian':4, 'crypto':5, 'electronics':6, 'forsale':7, 'graphics':8, 'guns':9,\
	 'hockey':10, 'mac':11, 'xwindows':12, 'windows':13, 'space':14, 'religion':15, 'politics':16, 'pc':17, 'motorcycles':18, 'mideast':19, 'medical':20}

#writes all sums - word frequencies corresponding to each word under a title, total words under each [opic and total words under all [opics 
def make_model(directory):
	print "Bayes"
	dir_list =  listdir(directory)
	total_words_under_all = 0
	for d in dir_list:
		freq_dict.update({d:{}})#initializing keys with empty values in the dictionary
		path = directory+"/"+d
		total_words = 0
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				calculate_from_file(d,path+"/"+name)  #total words in a particular file under the topic directory
			topic_documents.update({d : len(files)})

	return

#opens a file and reads it line by line, calculating and recording/updating word frequencies as it goes
def calculate_from_file(d, filepath):
	f = open(filepath, 'r')
	d = str(d)
	words_in_file = 0
	lowercaselist = []
	for line in f:
		cleanLine = re.sub('\W+',' ', line )
		wordlist = cleanLine.split()	#getting rid of all special chars 
		for x in wordlist:
			lowercaselist.append(x.lower())
	lowercaselist = list(set(lowercaselist))
	for x in lowercaselist:
		if x in freq_dict[d]:
			freq_dict[d][x] += 1
		else:
			freq_dict[d].update({x : 1})
	return

#revise model to include probabilities of words given title
def revise_model():
	for topic in freq_dict:
		for word in freq_dict[topic]:
			freq_dict[topic][word] = float(freq_dict[topic][word])/topic_documents[topic]
	return

#predicts the topic based on MLE							
def predict_topic(directory, loaded_model):
 	dir_list =  listdir(directory)
	topic_dict={} #keeps count of words predicted to belong to a certain topic in a certain file
	index_dict={} #indicating which index in the confusion matrix belongs to which topic
	count = 0 #just a counter to help in identifying indexes corresponding to topics in the confusion matrix
	w = 20 #length and width of the confusion matrix - given assumption in problem statement - total number of topics is 20 
	conf_mtr = [[0 for x in range(w)] for y in range(w)] #intializing confusion matrix	
	total = 0
	correct = 0
	del loaded_model['.DS_Store']

	for d in dir_list:
		path = directory+"/"+d
		topic_posterior = {}
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				for topic in loaded_model:
					f = open(path+"/"+name, 'r')
					temp = 0.0
					for line in f:	
						cleanLine = re.sub('\W+',' ', line ) #cleaning line to exclude special chars 
						wordlist = cleanLine.split() #getting proper words from the line
						for p in wordlist:
							if p in loaded_model[topic]:
								temp += math.log(loaded_model[topic][p])
							else:
								temp += math.log(0.0001)
					topic_posterior.update({topic : temp})
				# temp = sum(topic_posterior.values())
				# for key, value in topic_posterior.items():
				# 	topic_posterior[key] = float(value) / temp
				# topic_predicted = np.random.choice(topic_posterior.keys(), p = topic_posterior.values())
				topic_predicted = max(topic_posterior, key=topic_posterior.get)
				print "Topic predicted - ", topic_predicted
				if topic_predicted == d:
					correct += 1
				else: conf_mtr[Topics[d]-1][Topics[topic_predicted]-1]+=1 #"adding to the confusion"
				total += 1

	print "Accuracy: ",
	print float(correct)*100/total, "\n"	
	
	#printing confusion matrix
	for x in range(20):
		temp = ""
		for y in range(20):
			temp = temp + str(conf_mtr[x][y])+"\t"
		print temp+"\n"	
	return

def findTopTen():
	top_dict={}
	f = open('distinctive_words.txt', 'w')
	for key in freq_dict:
		if not key.startswith( 'total', 0, 5 ):
			top_ten = heapq.nlargest(10,freq_dict[key],freq_dict[key].get)
			top_dict.update({key:top_ten})
			#json.dump({key:top_ten}, f, separators=('\n', ': '))
			#json.dump("\n",f)
	json.dump(top_dict, f)
input = sys.argv[1:5]
if len(input) == 4:
	mode = input[0]
	directory = input[1]
	model_file = input[2]
	fraction = input[3]
else:
	print "enter all input parameters!"

print mode, directory, model_file, fraction
if mode == "train":
	make_model(directory)
	revise_model()
	findTopTen()
	pickle.dump(freq_dict, open(str(model_file), "wb" ) )	
#load_dict = pickle.load( open(str(model_file), "rb" ) )
#	print load_dict
if mode == "test":
	print "test"
	loaded_model = pickle.load( open(str(model_file), "rb" ) )
	predict_topic(directory,loaded_model)
		
