# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:38:41 2019

@author: Administrator
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

import socket
import struct

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

rule_number = 10000

def preprocess(path):
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
    head = tf.convert_to_tensor(head)

    index = tf.convert_to_tensor(A[:,6])
    
    return head,index



class classifier():
    def __init__(self,path):
        rule_set = Init_rule_set(path)
        self.rule = tf.convert_to_tensor(rule_set,dtype=tf.int64)
        self.mask = tf.convert_to_tensor([1,1,1,1,1,-1,-1,-1,-1,-1],dtype=tf.int64)
        
    def call(self,x):
        x = tf.reshape(x,[-1,1,5])
        x = tf.tile(x,[1,1,2])
        x = (x - self.rule)*self.mask
        x = (x >= 0)
        x = tf.reduce_all(x,axis=2)

#        x = tf.where(x)
#        x = tf.math.segment_min(x,x[:,0])[:,1]
        
        x = tf.cast(x,dtype=tf.int8)
        x = tf.argmax(x,axis=1)
        
        return x


if __name__ == "__main__":
    path = "data/rule_{0}.rule".format(rule_number)
    model = classifier(path)
    i = 1
    x,y = preprocess("data/rule_{0}_{1}.trace".format(rule_number,i))
    
    batch = 256
    st = 0
    start_time = time.time()
    while st < x.shape[0]:
        re = model.call(x[st:st+batch])
        st = st + batch
    print(time.time()-start_time)
    #print(tf.equal(re,y[0:64]))
