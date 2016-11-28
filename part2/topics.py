import sys, os, re, pickle,json,heapq
from os import listdir
from os.path import isfile, join, walk

freq_dict={}
total_topics = 20 #assuming as given in problem statement

#writes all sums - word frequencies corresponding to each word under a title, total words under each topic and total words under all topics 
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
				total_words = total_words + calculate_from_file(d,path+"/"+name)  #total words in a particular file under the topic directory
		freq_dict.update({"total words under "+d:total_words}) #updating the dictionary to reflect the total words under a particular topic
		total_words_under_all = total_words_under_all + total_words #total words under all files under a topic
	freq_dict.update({"total words under all": total_words_under_all}) #updating the dictionary to reflect total words under all titles

#opens a file and reads it line by line, calculating and recording/updating word frequencies as it goes
def calculate_from_file(d, filepath):
	f = open(filepath, 'r')
	d = str(d)
	words_in_file = 0
	for line in f:
		cleanLine = re.sub('\W+',' ', line )
		wordlist = cleanLine.split()	#getting rid of all special chars 
		lowercaselist = [x.lower() for x in wordlist]	#converting all proper words to lower case for uniformity
		wordfreq = [wordlist.count(p) for p in lowercaselist] #calculating word frequencies in the line 
    		freq_dict[d].update(dict(zip(lowercaselist,wordfreq)))	#updating frequencies in the dictionary
		words_in_file = words_in_file+sum(wordfreq)	#calculating total words in the file
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
        			for line in f:	
                			cleanLine = re.sub('\W+',' ', line ) #cleaning line to exclude special chars 
                			wordlist = cleanLine.split() #getting proper words from the line
					for p in wordlist:
						max_val = float(1)/10**50 #random value assumed for comparison
						topic=d
						for key in loaded_model:
                					if not key.startswith( 'total', 0, 5 ):
								if p.lower() in loaded_model[key]:
									if loaded_model[key][p.lower()] > max_val:
										max_val = loaded_model[key][p.lower()]
										topic = key				
						topic_dict[topic]= topic_dict[topic]+ 1
				max_cnt = -1
				topic = ""
				for d2 in dir_list:
					if topic_dict[d2] > max_cnt and topic_dict[d2] < 80:
						max_cnt = topic_dict[d2]
						topic = str(d2)
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
	findTopTen()
	pickle.dump(freq_dict, open(str(model_file), "wb" ) )	
#load_dict = pickle.load( open(str(model_file), "rb" ) )
#	print load_dict
if mode == "test":
	print "test"
	loaded_model = pickle.load( open(str(model_file), "rb" ) )
	predict_topic(directory,loaded_model)
		
