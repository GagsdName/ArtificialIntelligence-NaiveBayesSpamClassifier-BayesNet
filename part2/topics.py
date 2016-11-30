import sys, os, re, pickle,json,heapq
from os import listdir
from os.path import isfile, join, walk
import collections
import numpy as np

freq_dict=collections.defaultdict(list)
prob_dict = collections.defaultdict(dict)
total_topics = 20 #assuming as given in problem statement
all_topics = ['atheism', 'autos', 'baseball', 'christian', 'crypto', 'electronics', 'forsale', 'graphics', 'guns', 'hockey', 'mac', 'xwindows', 'windows', 'space', 'religion', 'politics', 'pc', 'motorcycles', 'mideast', 'medical']

#writes all sums - word frequencies corresponding to each word under a title, total words under each topic and total words under all topics 
def make_model(directory):
	print "Bayes"
	dir_list =  listdir(directory)
	total_words_under_all = 0
	for d in dir_list:
		path = directory+"/"+d
		total_words = 0
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				calculate_from_file(d,path+"/"+name)  #total words in a particular file under the topic directory
	
#opens a file and reads it line by line, calculating and recording/updating word frequencies as it goes
def calculate_from_file(d, filepath):
	f = open(filepath, 'r')
	d = str(d)
	words_in_file = 0
	for line in f:
		cleanLine = re.sub('\W+',' ', line )
		wordlist = cleanLine.split()	#getting rid of all special chars 
		lowercaselist = [x.lower() for x in wordlist]	#converting all proper words to lower case for uniformity
		for word in lowercaselist:
			freq_dict[word].append(d)
	return

#revise model to include probabilities of words given title
def revise_model():
	for key in freq_dict:
		temp = get_topics_list(freq_dict[key])
		prob_dict[key] = temp
	return

def get_topics_list(topics):
	topics_count_list = []
	topicscount = {}
	for topic in topics:
		if topic in topicscount:
			topicscount[topic] = topicscount[topic] + 1
		else:
			topicscount.update({topic:1})
	total_count = sum(topicscount.values())
	for key in topicscount:
		topicscount[key] = float(topicscount[key])/total_count
	return topicscount

#predicts the topic based on MLE							
def predict_topic(directory, loaded_model):
 	dir_list =  listdir(directory)
	topic_dict={} #keeps count of words predicted to belong to a certain topic in a certain file
	index_dict={} #indicating which index in the confusion matrix belongs to which topic
	count = 0 #just a counter to help in identifying indexes corresponding to topics in the confusion matrix
	w = 20 #length and width of the confusion matrix - given assumption in problem statement - total number of topics is 20 
	conf_mtr = [[0 for x in range(w)] for y in range(w)] #intializing confusion matrix	

	#intialising confusion matrix
	for dr in dir_list:
        	index_dict.update({dr:count})
		count+=1

        #traversing test data to predict topic
	for d in dir_list:
                path = directory+"/"+d
                for root,dirs,files in os.walk(path):
                        for name in files:
                                for d1 in dir_list:
					topic_dict.update({d1:0}) #updating the dictionary to reflect number of word matches to corresponding topics  
				print d,name
                                f = open(path+"/"+name, 'r')
                                topic_count = {}
        			for line in f:	
                			cleanLine = re.sub('\W+',' ', line ) #cleaning line to exclude special chars 
                			wordlist = cleanLine.split() #getting proper words from the line
					for p in wordlist:
						word_dict = loaded_model[p.lower()]
						topics_list = word_dict.keys()
						prob_dist = word_dict.values()
						if topics_list:
							random_topic = np.random.choice(topics_list, p = prob_dist) 
						else:
							random_topic = np.random.choice(all_topics)

						if random_topic in topic_count:
							topic_count[random_topic] = topic_count[random_topic]+1
						else:
							topic_count.update({random_topic:1})
				topic = max(topic_count, key=topic_count.get)
				print "Topic predicted - ", topic
				if topic != d:
					conf_mtr[index_dict[d]][index_dict[topic]]+=1 #"adding to the confusion"

	#printing confusion matrix
	for x in range(20):
		temp = ""
		for y in range(20):
			temp = temp + str(conf_mtr[x][y])+" "
		print temp+"\n"		
	
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
	# findTopTen()
	print prob_dict
	pickle.dump(prob_dict, open(str(model_file), "wb" ) )	
#load_dict = pickle.load( open(str(model_file), "rb" ) )
#	print load_dict
if mode == "test":
	print "test"
	loaded_model = pickle.load( open(str(model_file), "rb" ) )
	predict_topic(directory,loaded_model)
		
