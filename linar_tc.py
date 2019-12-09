# -*- coding: utf-8 -*-

import numpy as np
import time
import torch as tc

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
            
            



if __name__ == "__main__":
    rule_number = 256
    path = "data/rule_{0}.rule".format(rule_number)
    model = linar_classifier(path)
    
    i = 1
    x,y = model.preprocess("data/rule_{0}_{1}.trace".format(rule_number,i))
    
    start_time = time.time()
    re = model.call_on_batch(x,256)
    print(time.time()-start_time)


