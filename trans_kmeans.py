# -*- coding: utf-8 -*-

import numpy as np
import time
import torch as tc

import socket
import struct
import math


def ip2uint(ip):
  #Convert IP string int
  packedIP = socket.inet_aton(ip)
  return struct.unpack("!L", packedIP)[0]

class Rule:
    def __init__(self,line):
        L = list(line[1:].split())
        sip, smasklen = L[0].split('/')
        dip, dmasklen = L[1].split('/')
        self.sport_b, self.sport_e = int(L[2]), int(L[4])
        self.dport_b, self.dport_e = int(L[5]), int(L[7])
        prot , protmask = map(eval,L[8].split('/'))
        #self.prio0,self.prio1 = map(eval,L[9].split('/'))
        sip = ip2uint(sip)
        dip = ip2uint(dip)
        smask = ((1 << int(smasklen)) - 1) << (32 - int(smasklen))
        dmask = ((1 << int(dmasklen)) - 1) << (32 - int(dmasklen))
        self.sip_b = sip & smask
        self.sip_e = self.sip_b + (0xffffffff ^ smask)
        self.dip_b = dip & dmask
        self.dip_e = self.dip_b + (0xffffffff ^ dmask)
        self.prot_b = prot & protmask
        self.prot_e = self.prot_b + (0xff ^ protmask)

    def numpy(self):
        return np.array([self.sip_b,self.dip_b,self.sport_b,self.dport_b,
                         self.prot_b,self.sip_e,self.dip_e,
                         self.sport_e,self.dport_e,self.prot_e])


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




class linar_classifier():
    def __init__(self,path):
        rule_set = Init_rule_set(path)
        self.rule = tc.tensor(rule_set,
                              dtype=tc.int64,
                              device="cuda")
        self.mask = tc.tensor([1,1,1,1,1,-1,-1,-1,-1,-1],
                              dtype=tc.int64,
                              device="cuda")
        
    def preprocess(self,path):
        f = open(path)
        A = np.array(list(map(int,f.read().split())))
        A = A.reshape((-1,7))
        f.close() 
        sip = A[:,0].reshape((-1,1)) 
        dip = A[:,1].reshape((-1,1))
        sport = A[:,2].reshape((-1,1))
        dport = A[:,3].reshape((-1,1))
        prot = A[:,4].reshape((-1,1))
    
        head = np.concatenate((sip,dip,sport,dport,prot),axis=1)
        head = tc.tensor(head,device="cuda")
    
        index = tc.tensor(A[:,6],device="cuda")
        
        return head,index
        
    def call(self,x):
        x = tc.reshape(x,[-1,1,5])
        x = x.repeat([1,1,2])
        x = (x - self.rule)*self.mask
        x = (x >= 0)
        x = x.all(dim=2)
        x = tc.where(x,
                        tc.tensor(range(x.shape[1]),dtype=tc.int32,device="cuda"),
                        tc.tensor(x.shape[1],dtype=tc.int32,device="cuda"))
        x = x.argmin(dim=1)

        return x
    
    def call_on_batch(self,x,batch):
        size = x.shape[0]
        i = 0
        y = tc.zeros(size,dtype=tc.int64,device="cuda")
        while i < size:
            y[i:i+batch] = self.call(x[i:i+batch])
            i = i + batch
        return y
            
def batch_preprocess(x): # essentially the same as that in train.py
    
    A = x.cpu()
    
    sip = A[:,0].numpy()
    sip = np.array([sip&0xff,(sip&0xff00)>>8,(sip&0xff0000)>>16,(sip&0xff000000)>>24]).T
    
    dip = A[:,1].numpy()
    dip = np.array([dip&0xff,(dip&0xff00)>>8,(dip&0xff0000)>>16,(dip&0xff000000)>>24]).T
    
    sport = A[:,2].numpy()
    sport = np.array([sport&0xff,(sport&0xff00)>>8]).T
    dport = A[:,3].numpy()
    dport = np.array([dport&0xff,(dport&0xff00)>>8]).T
    prot = A[:,4].numpy().reshape((-1,1))
    
    head = np.concatenate((sip,dip,sport,dport,prot),axis=1)
    
#    head = tc.tensor(head,dtype=tc.int64).cuda()
#    head = tc.nn.functional.one_hot(head,256)
#    head = tc.reshape(head,(-1,256*13)).to(dtype=tc.float32,device="cuda")
    
    head = tc.tensor(head,dtype=tc.float32,device="cuda")
    head = head/128 - 1
    
    return head
#################  K-means++ Zone ##################

centroids = []

def euler_distance(p1, p2): 
    # calculate the distance of two tuples using the 3rd element of the tuple.
    distance = 0.0
    for i in range(5):
        distance += math.pow(p1[i]-p2[i], 2)
    return math.sqrt(distance)

def get_closest_dist(point):
    global centroids
    min_dist = math.inf
    for i, centr in enumerate(centroids):
        dist = euler_distance(centr, point[2])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def K_find_centers(num, rules):
    # get initial centroids.
    global centroids
    centroids.append(rules[np.random.randint(0, rule_number)][2])
    d = [0 for _ in range(rule_number)]
    for _ in range(1, num):
        print(_)
        total = 0.0
        for i, point in enumerate(rules):
            d[i] = get_closest_dist(point)
            total += d[i]
        total *= np.random.random()
        for i, di in enumerate(d):
            total -= di
            if total > 0:
                continue
            centroids.append(rules[i][2])
            break
    return centroids

from collections import defaultdict

def point_avg(points):
    dimensions = len(points[0])
    pts = len(points)
    new_center = []
    for dimension in range(dimensions):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]
        new_center.append(dim_sum/float(pts))
    return new_center

def update_centers(rules, assignments):
    new_means = defaultdict(list)
    centers = []
    for i, rule in zip(assignments, rules):
        if (i + 1) % 1000 == 0:print(i+1)
        new_means[i].append(rule[2])
    for points in new_means.values():
        centers.append(point_avg(points))
    return centers


def assign_points(rules):
    assignments = []
    cnt = 0
    for rule in rules:
        index, shortest = 0, math.inf
        cnt += 1
        if cnt % 1000 == 0: print(cnt)
        for i in range(len(centroids)):
            val = euler_distance(rule[2], centroids[i])#[2] is for getting point
            if val < shortest:
                shortest, index = val, i
        assignments.append(index)
    return assignments

def K_means(num, rules, prepared_centroids=None):
    global centroids
    if prepared_centroids != None: centroids = prepared_centroids
    else:
        centroids = []
        for i in range(num):
            centroids.append(rules[np.random.randint(0, rule_number)][2])
    old_assignments = None
    assignments = assign_points(rules)
    times = 0
    while assignments != old_assignments:
        times += 1
        print('iter', times)
        centroids = update_centers(rules, assignments)
        print("centers updated...")
        old_assignments = assignments
        assignments = assign_points(rules)
        print("points assigned...")
    return assignments
#####################################################
if __name__ == "__main__":
    rule_number = 100000
    path = "data/rule_{0}.rule".format(rule_number)
    rules, i = [], 0   
    for x in Init_rule_set(path):
        rules.append((x, i, ((x[0]+x[1])>>1,(x[2]+x[3])>>1,(x[4]+x[5])>>1,(x[6]+x[7])>>1,(x[8]+x[9])>>1)))
        # tuple with three elements: data itself, tag and center(used as "points" in K-means++)
        i += 1
    # K_find_centers(200, rules)
    rule_number = len(rules)
    assignments = K_means(200, rules)
    f=open("result.txt","w")
    for i, x in enumerate(assignments):
        f.write(str(i)+" "+str(x)+'\n')
    """ 
    rule_number = 100000
    for i in range(1,601):
        print("processing file #",i)
        x,y = model.preprocess("data/rule_{0}_{1}.trace".format(rule_number,i))
        print("headers num =",x.shape[0])
        x = batch_preprocess(x)
        f = open("data/processed_txts/rule_{0}_{1}.txt".format(rule_number,i),"w")
        for j in range(x.shape[0]):
            for eks in x[j, :]:
                f.write(str(eks.item())+" ") 
            f.write(str(y[j].item()//500)+"\n")
            f.flush()
            if j % 10000 == 0: print("line =",j)
        f.close()

    # start_time = time.time()
    # re = model.call_on_batch(x,256)
    print(time.time()-start_time)
    """

