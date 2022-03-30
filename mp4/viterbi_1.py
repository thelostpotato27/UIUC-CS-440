# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    import math
    words_set = set()
    tags_set = set()

    for sentence in train:
        for elem in sentence:
            words_set.add(elem[0])
            tags_set.add(elem[1])
    
    initial_tags = {}
    for sentence in train:
        if sentence[0][1] not in initial_tags:
            initial_tags[sentence[0][1]] = 0
        initial_tags[sentence[0][1]] += 1
    
    initial_prob = {}
    k = .0001
    for key in initial_tags:
        initial_prob[key] = math.log((initial_tags[key] + k)/(len(train) + k*len(tags_set)))

    transition_dict = {}

    for sentence in train:
        for elems in range(len(sentence) - 1):
            curr_tag = sentence[elems][1]
            next_tag = sentence[elems + 1][1]
            if curr_tag not in transition_dict:
                transition_dict[curr_tag] = {}
            if next_tag not in transition_dict[curr_tag]:
                transition_dict[curr_tag][next_tag] = 0
            transition_dict[curr_tag][next_tag] += 1
    

    new_tag_prob = math.log((k)/(len(train) + k*len(tags_set)))
    for curr_tag in tags_set:
        transition_amount = 0
        for next_tag in tags_set:
            if curr_tag in transition_dict and next_tag in transition_dict[curr_tag]:
                transition_amount += 1
        for next_tag in tags_set:
            if curr_tag in transition_dict and next_tag in transition_dict[curr_tag]:
                transition_dict[curr_tag][next_tag] = math.log((transition_dict[curr_tag][next_tag] + k)/(transition_amount + k*len(tags_set)))
            else:
                if curr_tag not in transition_dict:
                    transition_dict[curr_tag] = {}
                if next_tag not in transition_dict[curr_tag]:
                    transition_dict[curr_tag][next_tag] = new_tag_prob


    for sentence in test:
        matrix = []
        back_pointer = []
        for i in range(len(sentence)):
            matrix.append({tag:0 for tag in tags_set})
            back_pointer.append({tag:None for tag in tags_set})

        

        


    return []