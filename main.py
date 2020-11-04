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

def split_dataset(data,kfold = 10,shuffle = False,validation = False):
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
    if validation == 1:
        result['test_set'] = slices[-1]
        slices = slices[:-1]
    result['train_and_validation_set'] = slices
    return result
# #%%Usage
# split_datasets = split_dataset(data)
# train_and_validation = split_datasets['train_and_validation_set']
# test_set = split_datasets['test_set']
# for i in range(10):
#     for training_data in train_and_validation:
#         if training_data[-1] - find_label(tree,training_data) < 0.1:
#             correct+=1
#%%
import matplotlib.pyplot as plt

#定义文本框和箭头格式
decisionNode = dict(boxstyle='round', fc="1",)
leafNode = dict(boxstyle='round', fc='1')
arrow_args = dict(arrowstyle="-")

#绘制带箭头的注解 实际的绘图功能
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha='center', bbox=nodeType, arrowprops=arrow_args)

# def getNumLeafs_my(tree):
#     if not tree:
#         return 0
#     if isinstance(tree,float):
#         return 1
#     return getNumLeafs(tree["left"]) + getNumLeafs(tree["right"])

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
    
def getTreeDepth(myTree):
    return 4
    

    
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree) #计算宽
    depth = getTreeDepth(myTree) #计算高
    firstStr = list(myTree.keys())[0]
    #计算已经绘制的节点的位置，以及放置下一个节点的恰当位置
    #通过计算树所包含的所有叶子节点数，划分图形的宽度，从而计算得到当前节点的中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # 标记子节点的属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 按比例减少全局变量plotTree.yOff,并标注此处需要绘制子节点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #依次递减y坐标
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD #在绘制完所有子节点以后，增加全局变量Y的偏移值
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree)) #存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree)) #存储树的深度
    #追踪已经绘制的节点的位置
    plotTree.xOff = -0.5/plotTree.totalW 
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
#%%
a = {'tearRate': {'normal': {'astigmatic': {'yes': {'prescript': {'hyper': {'age': {'pre': 'no lenses', 'presbyopic': 'no lenses', 'young': 'hard'}}, 'myope': 'hard'}}, 'no': {'age': {'pre': 'soft', 'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}, 'young': 'soft'}}}}, 'reduced': 'no lenses'}}
createPlot(a)
# %%

# %%
