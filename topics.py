import sys, glob, os, re, pickle
from os import listdir
from os.path import isfile, join, walk

freq_dict={}

def calculate_Bayesian(directory):
	print "Bayes"
	dir_list =  listdir(directory)
	for d in dir_list:
		freq_dict.update({d:{}})
		path = directory+"/"+d
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				calculate_from_file(d,path+"/"+name)


def calculate_from_file(d, filepath):
	f = open(filepath, 'r')
	d = str(d)
	for line in f:
		cleanLine = re.sub('\W+',' ', line )
		wordlist = cleanLine.split()
		wordfreq = [wordlist.count(p) for p in wordlist]
    		freq_dict[d].update(dict(zip(wordlist,wordfreq)))
			
		
input = sys.argv[1:5]
if len(input) == 4:
	mode = input[0]
	directory = input[1]
	model_file = input[2]
	fraction = input[3]
else:
	print "enter all input parameters!"

print mode, directory, model_file, fraction
if mode == "training":
	calculate_Bayesian(directory)
	pickle.dump(freq_dict, open(str(model_file), "wb" ) )
	load_dict = pickle.load( open(str(model_file), "rb" ) )
if mode == "test":
	print "test"
