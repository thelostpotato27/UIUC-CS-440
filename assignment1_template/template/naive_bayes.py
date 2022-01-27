# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    
    # print(len(y),'y')
    pos_vocab = {}
    neg_vocab = {}

    for i in range(len(y)):
        if y[i]:
            for x in X[i]:
                if x in pos_vocab:
                    pos_vocab[x] += 1
                else:
                    pos_vocab[x] = 1
        else:
            for x in X[i]:
                if x in neg_vocab:
                    neg_vocab[x] += 1
                else:
                    neg_vocab[x] = 1
    # print(pos_vocab['the'])
    # ##TODO:
    # raise RuntimeError(print(len(X),'X'))
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    # print(X[0],'X')
    pos_vocab = {}
    neg_vocab = {}

    for i in range(len(y)):
        if y[i]:
            for x in X[i]:
                if x in pos_vocab:
                    pos_vocab[x] += 1
                else:
                    pos_vocab[x] = 1
            for j in range(1, len(X[i])):
                bi_group = X[i][j-1] + " " + X[i][j]
                if bi_group in pos_vocab:
                    pos_vocab[bi_group] += 1
                else:
                    pos_vocab[bi_group] = 1
        else:
            for x in X[i]:
                if x in neg_vocab:
                    neg_vocab[x] += 1
                else:
                    neg_vocab[x] = 1
            for j in range(2, len(X[i])):
                bi_group = X[i][j-1] + " " + X[i][j]
                if bi_group in neg_vocab:
                    neg_vocab[bi_group] += 1
                else:
                    neg_vocab[bi_group] = 1


    ##TODO:
    # raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    dev_labels = []
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels, max_size=None)

    pos_max = 0
    pos_key = 0
    for key in pos_vocab:
        pos_max += pos_vocab[key]
        pos_key += 1
    
    neg_max = 0
    neg_key = 0
    for key in neg_vocab:
        neg_max = neg_max + neg_vocab[key]
        neg_key += 1


    for i in range(len(dev_set)):
        email = dev_set[i]
        ham = math.log(pos_prior)
        spam = math.log(1-pos_prior)

        for x in email:
            temp_pos = 0
            temp_neg = 0
            if x in pos_vocab:
                temp_pos = pos_vocab[x]
            ham += math.log((temp_pos + laplace)/(pos_max + laplace*(1 + pos_key)))

            if x in neg_vocab:
                temp_neg = neg_vocab[x]
            spam += math.log((temp_neg + laplace)/(neg_max + laplace*(1 + neg_key)))


        if ham > spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    #raise RuntimeError("Replace this line with your code!")

    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    dev_labels = []
    pos_vocab, neg_vocab = create_word_maps_bi(train_set, train_labels, max_size=None)

    uni_pos_max = 0
    uni_pos_key = 0
    bi_pos_max = 0
    bi_pos_key = 0
    for key in pos_vocab:
        if ' ' in key:
            bi_pos_max += pos_vocab[key]
            bi_pos_key += 1
        else:
            uni_pos_max += pos_vocab[key]
            uni_pos_key += 1
    
    uni_neg_max = 0
    uni_neg_key = 0
    bi_neg_max = 0
    bi_neg_key = 0
    for key in neg_vocab:
        if ' ' in key:
            bi_neg_max += neg_vocab[key]
            bi_neg_key += 1
        else:
            uni_neg_max += neg_vocab[key]
            uni_neg_key += 1



    for i in range(len(dev_set)):
        email = dev_set[i]
        uni_ham = math.log(pos_prior)
        uni_spam = math.log(1-pos_prior)

        for x in email:
            temp_pos = 0
            temp_neg = 0
            if x in pos_vocab:
                temp_pos = pos_vocab[x]
            uni_ham += math.log((temp_pos + unigram_laplace)/(uni_pos_max + unigram_laplace*(1 + uni_pos_key)))

            if x in neg_vocab:
                temp_neg = neg_vocab[x]
            uni_spam += math.log((temp_neg + unigram_laplace)/(uni_neg_max + unigram_laplace*(1 + uni_neg_key)))
        
        uni_ham *= (1-bigram_lambda)
        uni_spam *= (1-bigram_lambda)


        bi_ham = math.log(pos_prior)
        bi_spam = math.log(1-pos_prior)

        for j in range(1,len(email)):
            bi_word = email[j-1] + " " + email[j]
            temp_pos = 0
            temp_neg = 0
            if bi_word in pos_vocab:
                temp_pos = pos_vocab[bi_word]
            bi_ham += math.log((temp_pos + bigram_laplace)/(uni_pos_max + bi_pos_max + bigram_laplace*(1 + uni_pos_key + bi_pos_key)))

            if bi_word in neg_vocab:
                temp_neg = neg_vocab[bi_word]
            bi_spam += math.log((temp_neg + bigram_laplace)/(uni_neg_max + bi_neg_max + bigram_laplace*(1 + uni_neg_key + bi_neg_key)))
        
        bi_ham *= bigram_lambda
        bi_spam *= bigram_lambda

        ham = uni_ham + bi_ham
        spam = uni_spam + bi_spam

        # print("spam",  spam)
        # print("ham",  ham)

        if ham > spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    # raise RuntimeError("Replace this line with your code!")

    return dev_labels
