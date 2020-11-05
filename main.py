#%%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def load_data(file):
    data = np.loadtxt(file) 
    return data

def decision_tree_learning(training_dataset, depth):
    # If all samples have the same label, then return
    labels = training_dataset[:, 7].tolist()
    labels_set = set(labels)
    if len(labels_set) == 1:
        leaf_node = labels[0]
        return leaf_node, depth
    # use node to represent a new decision tree with root as split value
    attribute, split_value = find_split(training_dataset) 
    node = {'attribute': attribute, 'value': split_value }    
    # split the dataset into two according to the split value
    left_ind = np.where(training_dataset[:, attribute] < split_value)[0]
    right_ind = np.where(training_dataset[:, attribute] >= split_value)[0]
    l_dataset = training_dataset[left_ind,:]
    r_dataset = training_dataset[right_ind,:]
    # continue to build the subtree by doing the recursion
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    node['left'] = l_branch
    node['right'] = r_branch
    
    return node, max(l_depth, r_depth)
    
   
def find_split(dataset):
    maxGain_list = [] # store the maximum information gain for each attribute
    split_value_list = [] # store the split value for maximum information gain for each attribute
    
    for attribute in range(0, 7):
        attr_values = dataset[:, attribute]
        S_all = dataset[attr_values.argsort()] # sort the value of the specified attribute 
        # generate the list of possible split value
        possible_split = []        
        for i in range(0, S_all.shape[0]-1):
            if(S_all[i, 7] != S_all[i+1, 7]):
                possible_split.append((S_all[i, attribute] + S_all[i+1, attribute])/2)                
        possible_split = list(set(possible_split)) # reduce the running time
        
        maxGain = float("-inf") # record the maximum information gain
        maxGain_split_value = float("-inf")  # record split value for the maximum information gain
        for t in possible_split:
            # split the set into two according to the split value
            S_left = S_all[np.where(S_all[:, attribute] < t)[0],:]
            S_right = S_all[np.where(S_all[:, attribute] >= t)[0],:]
            current_gain = Gain(S_all, S_left, S_right)
            # always store the maximum information gain and the corresponding split value
            if(current_gain > maxGain):
                maxGain = current_gain
                maxGain_split_value = t    
        maxGain_list.append(maxGain)
        split_value_list.append(maxGain_split_value)
    max_Gain = max(maxGain_list)
    best_attri = maxGain_list.index(max_Gain)
    best_split = split_value_list[best_attri]
    
    print('The attribute that generating the maximum IG is: ', best_attri, " with value: ", best_split)

    return best_attri, best_split
    

def Gain(S_all, S_left, S_right):
    return H(S_all) - Remainder(S_left, S_right)

# Calculate the information gain for a specified dataset
def H(dataset):
    label = dataset[:,7]
    total_count = label.shape[0]
    H = 0
    if(total_count!=0): 
        for label_value in [1,2,3,4]:
            label_count = str(label.tolist()).count(str(label_value))
            if(label_count>0):
                p = label_count/total_count
                H = H - p*np.log2(p)
    return H

def Remainder(S_left, S_right):
    left_count = S_left.shape[0]
    right_count = S_right.shape[0]
    Remainder = float(left_count) / (left_count+right_count) * H(S_left) + float(right_count) / (left_count+right_count) *  H(S_right)
    return Remainder
 
# find the label for test_data by using the trained_tree
def find_label(trained_tree, test_data):
    if not isinstance(trained_tree, dict):
        return trained_tree
    attr = trained_tree['attribute']
    val = trained_tree['value']
    if test_data[attr] < val:
        return find_label(trained_tree['left'], test_data)
    else:
        return find_label(trained_tree['right'], test_data)
    
  # load data  
clean_dataset = load_data("WIFI_db/clean_dataset.txt")
noisy_dataset = load_data("WIFI_db/noisy_dataset.txt")
print("load finished")

tree, depth = decision_tree_learning(clean_dataset, 0)


#%%
def cross_validation(data,kfold = 10,shuffle = False,validation = False):
    if shuffle != True and shuffle != False:
        raise ValueError("shuffle must be a Boolean Value")
    if validation != True and validation != False:
        raise ValueError("validation must be a Boolean Value")
    if not isinstance(kfold,int):
        raise ValueError("kfold must be an int ")
    result = {}
    if shuffle:
        np.random.shuffle(data)
    slices = np.split(data,kfold)
    all_test_db = []
    all_trained_tree = []
    for i in range(kfold):
        testing_set = slices[i]
        training_set = slices.copy()
        training_set.pop(i)
        training_set = np.vstack(training_set)
        all_test_db.append(testing_set)
        trained_tree, _ = decision_tree_learning(training_set, 0)
        all_trained_tree.append(trained_tree)
    return all_test_db,all_trained_tree


def evaluate(test_db, trained_tree):
    confusion_matrix = np.zeros((4,4))
    correct = 0
    wrong = 0
    for i in range(10):
        for r in test_db[i]:
            # print(r[7])
            actual = int(float(r[7]))-1
            predicted = int(float(find_label(trained_tree[i],r[:-1])))-1
            if actual == predicted:
                correct += 1
            else:
                wrong += 1
            confusion_matrix[actual][predicted] += 1
    print(confusion_matrix)
            # print(find_label(trained_tree[i],r))
    return correct / (correct + wrong)


# load data
clean_dataset = load_data("WIFI_db/clean_dataset.txt")
noisy_dataset = load_data("WIFI_db/noisy_dataset.txt")
print("load finished")
# print(clean_dataset)

tree, depth = decision_tree_learning(noisy_dataset, 0)
# test_db, trained_tree = cross_validation(noisy_dataset)
#evaluate(test_db, trained_tree)
'''
# Shuffle the dataset
np.random.shuffle(clean_dataset)
np.random.shuffle(noisy_dataset)
'''
'''
# simple test
correct = 0
for test_row_ind in range(2000):
    test_row = clean_dataset[test_row_ind, 0:7]
    pre_label = find_label(tree, test_row)
    ground_truth = clean_dataset[test_row_ind, 7]

    if(pre_label==ground_truth):
        correct += 1
        
print(correct/2000)
'''
#%%
def get_accuracy1(tree, dataset):
    test_db, trained_tree = cross_validation(dataset)
    return evaluate(test_db, trained_tree)
#%%
# slices = np.split(noisy_dataset,10)
# test = slices[-1]
# validation = slices[-2]
# training_set = np.hstack(slices[:8])
# tree,_ = decision_tree_learning(training_set,0)
#print(get_accuracy(tree,test))
def get_data(data,path):
    result = []
    temp = []
    for i in range(len(path)-1):
        if path[i]['left'] == path[i+1]:
            for j in data:
                if j[path[i]['attribute']] < path[i]['value']:
                    temp.append(j)
        if path[i]['right'] == path[i+1]:
            for j in data:
                if j[path[i]['attribute']] > path[i]['value']:
                    temp.append(j)
        data = temp
        temp = []
    return data
def get_most(data):
    result = {}
    for i in data:
        if int(i[-1]) in result.keys():
            result[int(i[-1])] += 1
        else:
            result[int(i[-1])] = 1
    print(result)
    return sorted(result.items(), key = lambda kv:(kv[1], kv[0]))[-1][0]
def get_accuracy(tree, dataset):
    correct = 0
    wrong = 0
    for i in dataset:
        if abs((find_label(tree,i)) - i[-1])  < 0.01:
            correct +=1
        else:
            wrong+=1
    return correct / (correct + wrong)
def repeat_Pruning(root,validation):
ori = root.copy()
first = Pruning(root,validation)
second = Pruning(first,validation)
if first == second:
    return first
else:
    return repeat_Pruning(second,validation)
#%%
def Pruning(tree, validation_set,curNode = None,path = None):
    print(curNode)
    print("")
    if not curNode:
        curNode = tree
    if not path:
        path = [curNode]
    else:
        path.append(curNode)
    print("path: " + str(len(path)))
    if isinstance(curNode,float) or isinstance(curNode,int):
        return curNode
    if isinstance(curNode['left'],float) and isinstance(curNode['right'],float):
        print("Removing")
        print(path[-1])
        print(path[-2])
        oriAcc = get_accuracy(tree,validation_set)
        print("oriAcc:" + str(oriAcc))
        oricurNode = curNode.copy()
        try:
            data = get_data(validation_set,path)
            label = get_most(data)
        except:
            return curNode
        if path[-2]['left'] == curNode:
            path[-2]['left'] = label
        else:
            path[-2]['right'] = label
        newAcc = get_accuracy(tree,validation_set)
        print("newAcc:" + str(newAcc))
        if  newAcc < oriAcc:
            return curNode
        else:
            return label
    curNode['left'] = Pruning (tree,validation_set,curNode['left'],path.copy())
    curNode['right'] = Pruning (tree,validation_set,curNode['right'],path.copy())
    return curNode
    
        


            
    


# %%
