'''
Created on 27-Nov-2016
'''
# This class is used to represent a node of a tree
class Node:
    def __init__(self, label):
        self.value = label
        self.left = None
        self.right = None
        
    def setLeftChild(self, node):
        self.left = node
    
    def setRightChild(self, node):
        self.right = node
    
    def printNode(self):
        print("({} {} {})".format(self.value, self.left.value, self.right.value))