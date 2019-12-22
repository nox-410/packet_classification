# -*- coding: utf-8 -*-

import numpy as np
import time
import torch as tc
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
from linar_tc import linar_classifier
from tree.hicuts import HiCuts
from tree.efficuts import EffiCuts
from tree.tree import *

def preprocess(A): # n * 6 array.

    sip = A[:,0]
    sip = np.array([sip&0xff,(sip&0xff00)>>8,(sip&0xff0000)>>16,(sip&0xff000000)>>24]).T
    
    dip = A[:,1]
    dip = np.array([dip&0xff,(dip&0xff00)>>8,(dip&0xff0000)>>16,(dip&0xff000000)>>24]).T
    
    sport = A[:,2]
    sport = np.array([sport&0xff,(sport&0xff00)>>8]).T
    dport = A[:,3]
    dport = np.array([dport&0xff,(dport&0xff00)>>8]).T
    prot = A[:,4].reshape((-1,1))
    
    head = np.concatenate((sip,dip,sport,dport,prot),axis=1)
    
#    head = tc.tensor(head,dtype=tc.int64).cuda()
#    head = tc.nn.functional.one_hot(head,256)
#    head = tc.reshape(head,(-1,256*13)).to(dtype=tc.float32,device="cuda")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = tc.tensor(head,dtype=tc.float32,device=dev)
    head = head/128 - 1
    index = tc.tensor(A[:,5],device=dev)
    
    return head,index

class HeaderData(tc.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        


class classifier(tc.nn.Module):
    def __init__(self,in_dim,rule_number):
        super(classifier,self).__init__()
        self.fc0 = tc.nn.Linear(in_dim,256) # 128 -- 64 -- 32 99.7%
        self.fc1 = tc.nn.Linear(256,128)
        self.fc2 = tc.nn.Linear(128,64)
        self.fc3 = tc.nn.Linear(64,rule_number)
        
        
    def forward(self,x):
        x = tc.nn.functional.relu(self.fc0(x))
        x = tc.nn.functional.relu(self.fc1(x))
        x = tc.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        #should use logic here
        
        return x


def Init_rule_set(path):
    f = open(path)
    rule_array = []
    while True:
        line = f.readline()
        if not line:
            break
        rule = Rule(line)
        rule_array.append(rule.numpy())
    rule_set = np.array(rule_array)
    f.close()
    return rule_set

mark, mark2 = None, None
nodecnt = 0
current_mark = 1
rulecnt = 0
def dfs(tree, x):
    global nodecnt, current_mark, rulecnt, mark
    nodecnt += 1
    if tree.is_leaf(x) is True:
        for rule in x.rules:
            print(rule.priority, end=" ")
        print("")
        for rule in x.rules:
            if mark[rule.priority] == 0:
                print(rule.priority, current_mark)
                mark[rule.priority] = current_mark
                rulecnt += 1
                if rulecnt == 500:# Remember to alter this line if you need other than 500*200 classification.
                    rulecnt, current_mark = 0, current_mark + 1
    for y in x.children:
        dfs(tree, y)
    return rulecnt
def dfs2(tree, x):
    global nodecnt, current_mark, rulecnt, mark2
    nodecnt += 1
    if tree.is_leaf(x) is True:
        for rule in x.rules:
            print(rule.priority, end=" ")
        print("")
        for rule in x.rules:
            if mark2[rule.priority] == 0:
                print(rule.priority, current_mark)
                mark2[rule.priority] = current_mark
                rulecnt += 1
                if rulecnt == 500:# Remember to alter this line if you need other than 500*200 classification.
                    rulecnt, current_mark = 0, current_mark + 1
    for y in x.children:
        dfs2(tree, y)
    return rulecnt

def run_hicuts(ruleset):
    global mark
    cuts = HiCuts(ruleset)
    print("ruleset_len=",len(ruleset))
    cuts.train()
    print("len_mark=",len(mark))
    dfs(cuts.tree, cuts.tree.root)
    for i in range(len(mark)):
        if mark[i] == 0:print(ruleset[i].priority, ruleset[i])

def run_efficuts(ruleset):
    global nodecnt, mark2
    cutt = EffiCuts(ruleset)
    cutt.train()
    for tree in cutt.trees:
        print("depth=", tree.depth, "rules=", len(tree.root.rules))
        nodecnt = 0
        dfs2(tree, tree.root)
        print("cnt=", nodecnt)
    #print(mark2)
    for i in range(len(mark2)):
        if mark2[i] == 0:print(ruleset[i].priority, ruleset[i])

if __name__ == "__main__":
    rule_number = 100000
    # model = classifier(13,rule_number).cuda()
    classifier = linar_classifier("data/rule_{0}.rule".format(rule_number))
    #  loss_fn = tc.nn.CrossEntropyLoss()
    mark = np.zeros(classifier.rule.shape[0])
    mark2 = np.zeros(classifier.rule.shape[0])
    ruleset = []
    print(classifier.rule.shape)
    for i in range(classifier.rule.shape[0]):# for compability between cutting algorithms and our code.
        tup = []
        for j in range(5):
            tup.append(classifier.rule[i][j].item())
            tup.append(classifier.rule[i][j+5].item()+1)
        ruleset.append(Rule(i, tuple(tup)))

    # print(ruleset)
    run_hicuts(ruleset)
    f=open("mark.txt","w")
    for i in range(len(mark)):
        f.write(str(mark))
    f.close()
    #run_efficuts(ruleset) # if you need efficuts, uncomment these lines.
    #f2=open("mark2.txt","w")
    #for i in range(len(mark2)):
    #    f2.write(str(mark))
    #f2.close()
    for i in range(1,31):
        x,_ = classifier.preprocess("data/rule_{0}_{1}.trace".format(rule_number,i))
        print("read complete")
        y = classifier.call_on_batch(x,256)
        print(y)
        #save labeled(not relabeled) results.
        f=open("data/rule_{0}_{1}_labeled.trace".format(rule_number,i),"w")
        for xx, yy in zip(x, y):
            for xxx in xx:
                f.write(str(xxx.item())+" ")
            f.write(str(yy.item())+"\n")
        f.close()
        A = np.zeros((len(y), 6)).astype(np.int64)
        A[:,0:5] = x.cpu()
        A[:,5] = y.cpu()
        x, y = preprocess(A)
        f1=open("data/rule_{0}_{1}_coarse_labeled_notree.trace".format(rule_number,i),"w")
        for j in range(len(y)):
            if j % 10000 == 0:print(j)
            for k in range(13):
                f1.write(str(x[j, k].item())+" ")
            f1.write(str(y[j].item()//500)+"\n")# Alter this "500" if you need other than 200-classification for 100k rules.
        f1.close()
        f2=open("data/rule_{0}_{1}_coarse_labeled_hicuts.trace".format(rule_number,i),"w")
        for j in range(len(y)):
            if j % 10000 == 0:print(j)
            for k in range(13):
                f2.write(str(x[j, k].item())+" ")
            if mark[y[j]]>0: f2.write(str(int(mark[y[j]]-1))+"\n")
            else: f2.write("error!\n")# should not happen; these rules are covered by others.
        f2.close()
        # If you need efficuts, uncomment the following.
        """ 
        f3=open("data/rule_{0}_{1}_coarse_labeled_efficuts.trace".format(rule_number,i),"w")
        for j in range(len(y)):
            if j % 10000 == 0:print(j)
            for k in range(13):
                f3.write(str(x[j, k].item())+" ")
            if mark2[y[j]]>0: f3.write(str(int(mark2[y[j]]-1))+"\n")
            else: f3.write("error!\n")# should not happen; these rules are covered by others.
        f3.close()
        """
