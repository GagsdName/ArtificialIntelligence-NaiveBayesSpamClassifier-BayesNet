import sys, glob, os, re, pickle
from os import listdir
from os.path import isfile, join, walk
from decimal import Decimal

freq_dict={}
total_topics = 20 #assuming as given in problem statement

#writes all sums - word frequencies corresponding to each word under a title, total words under each topic and total words under all topics 
def make_model(directory):
	print "Bayes"
	dir_list =  listdir(directory)
	total_words_under_all = 0
	for d in dir_list:
		freq_dict.update({d:{}})
		path = directory+"/"+d
		total_words = 0
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				total_words = total_words + calculate_from_file(d,path+"/"+name)
		freq_dict.update({"total words under "+d:total_words})
		total_words_under_all = total_words_under_all + total_words
	freq_dict.update({"total words under all": total_words_under_all})

#opens a file and reads it line by line, calculating and recording/updating word frequencies as it goes
def calculate_from_file(d, filepath):
	f = open(filepath, 'r')
	d = str(d)
	words_in_file = 0
	for line in f:
		cleanLine = re.sub('\W+',' ', line )
		wordlist = cleanLine.split()
		wordfreq = [wordlist.count(p) for p in wordlist]
    		freq_dict[d].update(dict(zip(wordlist,wordfreq)))
		words_in_file = words_in_file+sum(wordfreq)
	return words_in_file
#revise model to include probabilities of words given title
def revise_model():
	for key in freq_dict:
		if not key.startswith( 'total', 0, 5 ):
			for w in freq_dict[key]:
				if freq_dict[key][w] > 0:
					prob = float(float(float(freq_dict[key][w])/freq_dict["total words under "+key] )* 1/20)\
						/ float(freq_dict[key][w])/freq_dict["total words under all"]
					freq_dict[key][w] = prob
#predicts the topic based on MLE							
def predict_topic(directory, loaded_model):
 	dir_list =  listdir(directory)
	topic_dict={}
        for d in dir_list:
                path = directory+"/"+d
                for root,dirs,files in os.walk(path):
                        for name in files:
                                for d1 in dir_list:
					topic_dict.update({d1:0})
				print d,name
                                f = open(path+"/"+name, 'r')
        			for line in f:	
                			cleanLine = re.sub('\W+',' ', line )
                			wordlist = cleanLine.split()
					for p in wordlist:
						max_val = float(1)/10000000000000000000000000000000000000000000000000000000000000000
						topic=d
						for key in loaded_model:
                					if not key.startswith( 'total', 0, 5 ):
								if p in loaded_model[key]:
									if loaded_model[key][p] > max_val:
										max_val = loaded_model[key][p]
										topic = key				
						topic_dict[topic]= topic_dict[topic]+ 1
				max_cnt = -1
				topic = ""
				for d2 in dir_list:
					if topic_dict[d2] > max_cnt and topic_dict[d2] < 100:
						max_cnt = topic_dict[d2]
						topic = str(d2)
				print "Topic predicted - ", topic	
			
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
	pickle.dump(freq_dict, open(str(model_file), "wb" ) )	
#load_dict = pickle.load( open(str(model_file), "rb" ) )
#	print load_dict
if mode == "test":
	print "test"
	loaded_model = pickle.load( open(str(model_file), "rb" ) )
	predict_topic(directory,loaded_model)
		
