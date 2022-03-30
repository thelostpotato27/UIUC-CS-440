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
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_dict = {}
    tag_dict = {}
    for sentence in train:
        for elem in sentence:
            if elem[0] not in word_dict:
                word_dict[elem[0]] = {}
            if elem[1] not in word_dict[elem[0]]:
                word_dict[elem[0]][elem[1]] = 0
            word_dict[elem[0]][elem[1]] += 1
            if elem[1] not in tag_dict:
                tag_dict[elem[1]] = 0
            tag_dict[elem[1]] += 1


    test_output = []

    max_tag_dict = ["null", -1]
    for tag in tag_dict:
        if tag_dict[tag] > max_tag_dict[1]:
            max_tag_dict = [tag, tag_dict[tag]]
    

    for sentence in test:
        prediction = []
        for elem in sentence:
            if elem in word_dict:
                tags = word_dict[elem].keys()
                max_tag = ["null", -1]
                for tag in tags:
                    if word_dict[elem][tag] > max_tag[1]:
                        max_tag = [tag, word_dict[elem][tag]]
                prediction.append((elem, max_tag[0]))
            else:
                prediction.append((elem, max_tag_dict[0]))
        # print(prediction)
        test_output.append(prediction)

    return test_output