# Possible template to start TP5

"""Compute the entropy of different models for text
            
Usage: compress [-m <model>] [-f <file>] [-o <order>]

Options:
-h --help      Show the description of the program
-f <file> --filename <file>  filename of the text to compress [default: Dostoevsky.txt]
-o <order> --order <order>  order of the model
-m <model> --model <model>  model for compression [default: IIDModel]
"""

import argparse, re
import numpy as np
import math
from collections import Counter 
import random
import os

from docopt import docopt

os.chdir('/Users/bastienbrier/Documents/TD5/')
#text = preprocess(open('Dostoevsky.txt').read())


list_of_symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                   'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '.', ' ', ',', ';']

class IIDModel:
    """An interface for the text model"""
    def __init__(self, order, src_text):
        print("Creation of the model")
        self.order = order
        # ...
        self.src_text = src_text
        
    def process(self,text):
        # ...
        symbols_dict = {}
        text = preprocess(text)
        for i in range(len(text) - self.order):
            symbol = text[i:i+self.order] # take the symbol of order nb of letters
            if symbol in symbols_dict: # if it exists, increment by 1
                symbols_dict[symbol] += 1
            else: # if it does not, create it
                symbols_dict[symbol] = 1
        return symbols_dict

    def getEntropy(self, text):
        # ...
        entropy = 0
        symbols_dict = self.process(text)
        total = sum(symbols_dict.values()) # number of total symbols
        for key in symbols_dict:
            prob = float(symbols_dict[key]) / total # number of key symbols over total
            entropy -= prob * np.log2(prob) # entropy formula
        return entropy

    def getCrossEntropy(self, text):
        # ...
        cross_entropy = 0
        KL = 0
        train_dict = self.process(self.src_text) # train dictionary
        target_entropy = self.getEntropy(text)
        target_dict = self.process(text) # target dictionary
        for key in target_dict:
            p = float(target_dict[key]) / sum(target_dict.values()) # target probability
            if key in train_dict: # handles 0 values in train
                q = float(train_dict[key]) / sum(train_dict.values()) # train probability
                KL += p * np.log2(p / q)
            else:
                KL += p * np.log2(p / 0.00000001)
                
        cross_entropy = target_entropy + KL
        return cross_entropy
    
    def generateSentence(self, length):
        length = length / self.order # select x symbols of length order
        symbols_dict = self.process(self.src_text)
        init_dict = symbols_dict.copy() # copy dictionary for initialization
        if length < 4: # too long after, and the characters are more likely to appear
            for i in range(len(symbols_dict.keys())): # we want to avoid beginning by these characters
                if (" " or "." or ";" or "," or "'\'") in symbols_dict.keys()[i]:
                    init_dict.pop(symbols_dict.keys()[i], None)
        
        count_vect = []
        total_dict = 0
        sentence = ''
        
        # Initialize the first symbol by probability
        total_dict = sum(init_dict.values())
        for key in init_dict:
            count_vect.append(init_dict[key]) # count vectors of the keys

        count_vect = [float(i) / total_dict for i in count_vect] # from count to prob
        prob_vect = []
        cumulative_prob = 0
        for i in range(len(count_vect)): # cumulative probability vector
            cumulative_prob += count_vect[i]
            prob_vect.append(cumulative_prob) 
        del cumulative_prob
        
        index_1 = 0
        threshold = random.random()
        for i in range(len(prob_vect)):
            if prob_vect[i] > threshold: # choose character with probability associated
                index_1 = i
                break
        sentence += init_dict.keys()[index_1] # first character of the sentence
        del count_vect
        del prob_vect
        
        # Generate next characters
        for i in range(self.order, length):
            count_vect = []
            total_dict = sum(symbols_dict.values())
            
            for key in symbols_dict:
                count_vect.append(symbols_dict[key]) # count vectors of the keys of the dict

            count_vect = [float(i) / total_dict for i in count_vect] # from count to prob
            prob_vect = []
            cumulative_prob = 0
            for i in range(len(count_vect)): # cumulative probability vector
                cumulative_prob += count_vect[i]
                prob_vect.append(cumulative_prob) 
            del cumulative_prob
        
            index_1 = 0
            threshold = random.random()
            for i in range(len(prob_vect)):
                if prob_vect[i] > threshold:
                    index_1 = i
                    break
            sentence += symbols_dict.keys()[index_1] # first character of the sequence

        return sentence
        

class MarkovModel:
    """An interface for the text model"""
    def __init__(self, order, src_text):
        print("Creation of the model")
        self.order = order
        # ...
        self.src_text = src_text

    def process(self, text):
        # ...
        symbols_dict = {}
        precedent_dict = {}
        text = preprocess(text)
        
        # Dict of order length
        for i in range(len(text) - self.order):
            symbol = text[i:i+self.order] # take the symbol of order nb of letters
            if symbol in symbols_dict: # if it exists, increment by 1
                symbols_dict[symbol] += 1
            else: # if it does not, create it
                symbols_dict[symbol] = 1
        
        # Dict of order-1 length
        for i in range(len(text) - self.order):
            symbol = text[i:i+self.order-1] # take the symbol of order nb of letters
            if symbol in precedent_dict: # if it exists, increment by 1
                precedent_dict[symbol] += 1
            else: # if it does not, create it
                precedent_dict[symbol] = 1
        
        return (symbols_dict, precedent_dict)

    def getEntropy(self, text):
        # ...
        entropy = 0
        symbols_dict = self.process(text)[0]
        precedent_dict = self.process(text)[1]
        total = sum(symbols_dict.values()) # number of total symbols
        for key in symbols_dict:
            symbol = str(key)
            precedent = symbol[:self.order-1]
            prob_cond = float(symbols_dict[key]) / float(precedent_dict[precedent]) # conditional probability of order o knowing o-1
            prob = float(symbols_dict[key]) / total # number of key symbols over total
            entropy -= prob * np.log2(prob_cond) # entropy formula
        return entropy

    def getCrossEntropy(self, text):
        cross_entropy = 0
        KL = 0
        train_dict = self.process(self.src_text)[0] # train dictionary
        train_prec = self.process(self.src_text)[1] # train dictionary of order-1 values
        target_entropy = self.getEntropy(text)
        target_dict = self.process(text)[0] # target dictionary
        target_prec = self.process(text)[1] # target dictionary of order-1 values
        target_total = sum(target_dict.values()) # number of total symbols
        for key in target_dict:
            symbol = str(key)
            precedent = symbol[:self.order-1]
            p_cond = float(target_dict[key]) / float(target_prec[precedent]) # probability of symbol knowing the first order-1 letters
            p_mutual = float(target_dict[key]) / target_total # probability of symbol occuring
            if key in train_dict: 
                q_cond = float(train_dict[key]) / float(train_prec[precedent]) # probability of symbol knowing the first order-1 letters in train
                KL += p_mutual * np.log2(p_cond / q_cond)
            else: # handles 0 values in train
                KL += p_mutual * np.log2(p_cond / 0.00000001)
                
        cross_entropy = target_entropy + KL
        return (cross_entropy, KL)
    
    def generateSentence(self, length):
        symbols_dict = self.process(self.src_text)[0]
        init_dict = symbols_dict.copy() # copy dictionary for initialization
        if length < 4: # too long after, and the characters are more likely to appear
            for i in range(len(symbols_dict.keys())): # we want to avoid beginning by these characters
                if (" " or "." or ";" or "," or "'\'") in symbols_dict.keys()[i]:
                    init_dict.pop(symbols_dict.keys()[i], None)
        
        count_vect = []
        total_dict = 0
        sentence = ''
        
        # Initialize the first symbol by probability
        total_dict = sum(init_dict.values())
        for key in init_dict:
            count_vect.append(init_dict[key]) # count vectors of the keys

        count_vect = [float(i) / total_dict for i in count_vect] # from count to prob
        prob_vect = []
        cumulative_prob = 0
        for i in range(len(count_vect)): # cumulative probability vector
            cumulative_prob += count_vect[i]
            prob_vect.append(cumulative_prob) 
        del cumulative_prob
        
        index_1 = 0
        threshold = random.random()
        for i in range(len(prob_vect)):
            if prob_vect[i] > threshold: # choose character with probability associated
                index_1 = i
                break
        sentence += init_dict.keys()[index_1] # first character of the sentence
        del count_vect
        del prob_vect
        
        # Generate next characters
        for i in range(self.order, length):
            prev_symbol = sentence[i-self.order+1:i]
            prec_list = []
            prec_val = []
            
            for key in symbols_dict:
                if prev_symbol == key[:len(key)-1]:
                    prec_list.append(key[len(key)-1:len(key)]) # if key corresponds to the last order-1 letters, add the last letter associated
                    prec_val.append(symbols_dict[key]) # add the value associated
            
            prec_val = [float(i) / sum(prec_val) for i in prec_val] # from count to prob
            
            prob_vect = []
            cumulative_prob = 0
            for i in range(len(prec_val)): # cumulative probability vector
                cumulative_prob += prec_val[i]
                prob_vect.append(cumulative_prob) 
            del cumulative_prob
            
            index_1 = 0
            threshold = random.random()
            for i in range(len(prob_vect)):
                if prob_vect[i] > threshold: # choose character with probability associated
                    index_1 = i
                    break
            sentence += prec_list[index_1] # first character of the sentence
            del prec_val
            del prec_list
            del prob_vect
            
        return sentence


def preprocess(text):
    text = re.sub("\s\s+", " ", text)
    text = re.sub("\n", " ", text)
    return text

# Experiencing encoding issues due to UTF8 (on possibly other texts)? Consider:
#  f.read().decode('utf8')
#  blabla.join(u'dgfg')
#              ^


if __name__ == '__main__':

    # Retrieve the arguments from the command-line
    args = docopt(__doc__)
    print(args)

    # Read and preprocess the text
    src_text = preprocess(open(args["--filename"]).read())
    goethe = preprocess(open("Goethe.txt").read())
    alighieri = preprocess(open("Alighieri.txt").read())
    hamlet = preprocess(open("Hamlet.txt").read())
    dostoevsky = preprocess(open("Dostoevsky.txt").read())

    # Create the model
    if(args["--model"]=="IIDModel"):
        model = IIDModel(int(args["--order"]), src_text)
    elif(args["--model"]=="MarkovModel"):
        model = MarkovModel(int(args["--order"]), src_text)

    model.process(src_text)
    print 'Entropy: ' + str(model.getEntropy(src_text))
    print 'Cross-Entropy Goethe: ' + str(model.getCrossEntropy(goethe)[0]) + '. KL: ' + str(model.getCrossEntropy(goethe)[1])
    print 'Cross-Entropy Alighieri: ' + str(model.getCrossEntropy(alighieri)[0]) + '. KL: ' + str(model.getCrossEntropy(alighieri)[1])
    print 'Cross-Entropy Hamlet: ' + str(model.getCrossEntropy(hamlet)[0]) + '. KL: ' + str(model.getCrossEntropy(hamlet)[1])
    print 'Cross-Entropy Dostoevsky: ' + str(model.getCrossEntropy(dostoevsky)[0]) + '. KL: ' + str(model.getCrossEntropy(dostoevsky)[1])
    print 'Sentence: ' + model.generateSentence(200)
