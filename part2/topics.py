import sys, os, re, pickle,json,heapq, random
from os import listdir
from os.path import isfile, join, walk
import numpy as np
import math

freq_dict={} #Contains entries in the format : {<topic>:{<word1>:<conditional probability>, <word2>:<conditional probability>}} 
topic_documents = {} #Maintains the total number of files in corresponding topics in the format: {<topio>:<file_count>}
total_topics = 20 #assuming as given in problem statement

#Topics and corresponding indexes are stored in this dictionary to help with creation of confusion matrix
Topics = {}

# helps write all sums - word frequencies corresponding to each word under a title in freq_dict
def make_model(directory, fraction):
	dir_list =  listdir(directory)#list of directories/topics
	for d in dir_list:
        	if not d.startswith('.'):
                	freq_dict.update({d:{}})#initializing keys with empty values in the dictionary

	for d in dir_list:
		path = directory+"/"+d #path to directory 
		for root,dirs,files in os.walk(path):
			for name in files:
				r = float(random.random())
				if r < fraction or r == fraction:
					print d,name
					calculate_from_file(d,path+"/"+name)#calculate frequency of words in the file 
				else:
					random_topic = np.random.choice(dir_list)
					while(random_topic.startswith('.')):
						random_topic = np.random.choice(dir_list)
					calculate_from_file(random_topic, path+"/"+name)
			topic_documents.update({d : len(files)}) #Updating topic_documents dictionary to indicate number of files under a particular topic

	return

#opens a file and reads it line by line, calculating and recording/updating word frequencies as it goes
def calculate_from_file(d, filepath):
	f = open(filepath, 'r')
	d = str(d)
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
	count = 0 #just a counter to help in identifying indexes corresponding to topics in the confusion matrix
	w = 20 #length and width of the confusion matrix - given assumption in problem statement - total number of topics is 20 
	conf_mtr = [[0 for x in range(w)] for y in range(w)] #intializing confusion matrix	
	total = 0 #total number of files predicted
	correct = 0 #correct number of files predicted
	#del loaded_model['.DS_Store']
	
	#assigning indexes to topics to help later in construction of confusion matrix	
	for d in dir_list:
		Topics.update({d:count})
		count+=1	
	
	for d in dir_list:
		path = directory+"/"+d
		topic_posterior = {} #holds log of topic posteriors for each topic
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				for topic in loaded_model:
					f = open(path+"/"+name, 'r')
					temp = 0.0 #intermediate likelihoods
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
				else: conf_mtr[Topics[d]-1][Topics[topic_predicted]-1]+=1 #"adding to the confusion" matrix
				total += 1
	
	#printing confusion matrix
	for x in range(20):
		temp = ""
		for y in range(20):
			temp = temp + str(conf_mtr[x][y])+"\t"
		print temp+"\n"	
	
	print "Confusion Matrix Index glossary",Topics
	
	print "Accuracy: ", float(correct)*100/total, "\n"
	
	return

#find top ten ocurring words under all topics
def findTopTen():
	top_dict={}
	f = open('distinctive_words.txt', 'w')
	for key in freq_dict:
		if not key.startswith( 'total', 0, 5 ):
			top_ten = heapq.nlargest(10,freq_dict[key],freq_dict[key].get)
			top_dict.update({key:top_ten})
			#json.dump({key:top_ten}, f, separators=('\n', ': '))
			#json.dump("\n",f)
	json.dump(top_dict, f) #writing output to distinctive_words.txt in json format

input = sys.argv[1:5] #input arguments
if len(input) == 4: #check to see if correct number of arguments are there
	mode = input[0]
	directory = input[1]
	model_file = input[2]
	fraction = float(input[3])
else:
	print "enter all input parameters!"


if mode == "train":
	make_model(directory, fraction) #call to make frequency dictionary - freq_dict
	revise_model() #revising freq_dict to indicated conditional probabilities of words given topic
	findTopTen() #find top ten words under each topic
	pickle.dump(freq_dict, open(str(model_file), "wb" ) ) #saving calculated model in the file as indicated in command arguments	

if mode == "test":
	loaded_model = pickle.load( open(str(model_file), "rb" ) ) #load saved model
	predict_topic(directory,loaded_model) #predict topic for each file, print them, print accuracy and print confusion matrix
		
