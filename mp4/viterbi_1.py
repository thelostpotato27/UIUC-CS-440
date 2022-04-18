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
    
    initial_tags = {}                       #initial prob
    for sentence in train:
        if sentence[0][1] not in initial_tags:
            initial_tags[sentence[0][1]] = 0
        initial_tags[sentence[0][1]] += 1
    
    initial_prob = {}
    k = .000001
    for key in initial_tags:
        initial_prob[key] = math.log((initial_tags[key] + k)/(len(train) + k*len(tags_set)))

    transition_dict = {}

    for sentence in train:                  #transition prob
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
                transition_amount += transition_dict[curr_tag][next_tag]
        for next_tag in tags_set:
            if curr_tag in transition_dict and next_tag in transition_dict[curr_tag]:
                transition_dict[curr_tag][next_tag] = math.log((transition_dict[curr_tag][next_tag] + k)/(transition_amount + k*len(tags_set)))
            else:
                if curr_tag not in transition_dict:
                    transition_dict[curr_tag] = {}
                if next_tag not in transition_dict[curr_tag]:
                    transition_dict[curr_tag][next_tag] = new_tag_prob


    emmission_dict = {}         #emission calc
    tag_dict_counter = {}

    for sentence in train:
        for elem in sentence:
            if elem[0] not in emmission_dict:
                emmission_dict[elem[0]] = {}
            if elem[1] not in emmission_dict[elem[0]]:
                emmission_dict[elem[0]][elem[1]] = 0
            emmission_dict[elem[0]][elem[1]] += 1
            if elem[1] not in tag_dict_counter:
                tag_dict_counter[elem[1]] = 0
            tag_dict_counter[elem[1]] += 1
    

    new_emission_prob = math.log((k)/(len(train) + k*(len(words_set)+1)))
    for tag in tags_set:
        for word in words_set:
            if word in emmission_dict:
                if tag in emmission_dict[word]:
                    emmission_dict[word][tag] = math.log((emmission_dict[word][tag] + k)/(tag_dict_counter[tag]+k*(len(words_set)+1)))          #maybe add 1?
                else:
                    emmission_dict[word][tag] = new_emission_prob
            else:
                emmission_dict[word][tag] = new_emission_prob

    predictions = [0] * len(test)
    counter = 0
    for sentence in test:
        conversion_matrix = []
        back_pointer = []
        for i in range(len(sentence)):
            conversion_matrix.append({tag:0 for tag in tags_set})
            back_pointer.append({tag:None for tag in tags_set})
        
        for key in conversion_matrix[0].keys():
            init_prob_offset = new_tag_prob
            emmission_prob = new_emission_prob
            if key in initial_prob:
                init_prob_offset = initial_prob[key]
            if sentence[0] in emmission_dict:
                if key in emmission_dict[sentence[0]]:
                    emmission_prob = emmission_dict[sentence[0]][key]
            conversion_matrix[0][key] = init_prob_offset + emmission_prob
        best_key = ["", -math.inf]
        for i in range(1, len(conversion_matrix)):
            for key in conversion_matrix[i].keys():
                best_key = ["", -math.inf]
                emmission_prob = new_emission_prob
                if sentence[i] in emmission_dict:
                    if key in emmission_dict[sentence[i]]:
                        emmission_prob = emmission_dict[sentence[i]][key]
                for prev_key in conversion_matrix[i-1].keys():
                    transition_prob = new_tag_prob
                    if prev_key in transition_dict:
                        if key in transition_dict[prev_key]:
                            transition_prob = transition_dict[prev_key][key]
                    # print("Key Candidate Key Candidate Key Candidate Key Candidate Key Candidate Key Candidate ")
                    # print(emmission_prob + transition_prob + conversion_matrix[i-1][prev_key])
                    # print("Key Candidate Key Candidate Key Candidate Key Candidate Key Candidate Key Candidate ")
                    if best_key[1] < emmission_prob + transition_prob + conversion_matrix[i-1][prev_key]:
                        best_key[1] = emmission_prob + transition_prob + conversion_matrix[i-1][prev_key]
                        best_key[0] = prev_key
                conversion_matrix[i][key] = best_key[1]
                back_pointer[i][key] = best_key[0]
        solution = []
        curr_index = len(conversion_matrix) - 1
        # print("Best Key Best Key Best Key Best Key Best Key Best Key Best Key ")
        # print(best_key)
        # print("Best Key Best Key Best Key Best Key Best Key Best Key Best Key ")
        key = best_key[0]
        # print(back_pointer)
        while curr_index >= 0:
            solution = [(sentence[curr_index], key)] + solution
            # print(back_pointer[curr_index])
            # print(key)
            # print("========================================")
            key = back_pointer[curr_index][key]
            curr_index -= 1
        predictions[counter] = solution
        counter += 1
    return predictions