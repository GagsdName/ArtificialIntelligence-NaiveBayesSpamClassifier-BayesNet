import sys, glob, os
from os import listdir
from os.path import isfile, join, walk

def calculate_Bayesian(directory):
	print "Bayes"
	dir_list =  listdir(directory)
	for d in dir_list:
		path = directory+"/"+d
		for root,dirs,files in os.walk(path):
			for name in files:
				print d,name
				calculate_from_file(path+"/"+name)


def calculate_from_file(filepath):
	f = open(filepath, 'r')
	for line in f:
		print line	
		
print "gagan"
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
if mode == "test":
	print "test"
