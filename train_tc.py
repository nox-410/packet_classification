# -*- coding: utf-8 -*-

import numpy as np
import time
import torch as tc
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
from linar_tc import linar_classifier


def preprocess(path):
    f = open(path)
    
    A = np.array(list(map(int,f.read().split())))
    A = A.reshape((-1,7))
    f.close()
    
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
    
    head = tc.tensor(head,dtype=tc.float32,device="cuda")
    head = head/128 - 1
    index = tc.tensor(A[:,6],device="cuda")
    
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
        self.fc0 = tc.nn.Linear(in_dim,256)
        self.fc1 = tc.nn.Linear(256,256)
        self.fc2 = tc.nn.Linear(256,256)
        self.fc3 = tc.nn.Linear(256,rule_number)
        
        
    def forward(self,x):
        x = tc.nn.functional.relu(self.fc0(x))
        x = tc.nn.functional.relu(self.fc1(x))
        x = tc.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        #should use logic here
        
        return x




if __name__ == "__main__":
    rule_number = 256
    
    model = classifier(13,rule_number).cuda()
    classifier = linar_classifier("data/rule_{0}.rule".format(rule_number))
    loss_fn = tc.nn.CrossEntropyLoss()
    
    
    def train_on_file(st,ed,lr):
        optimizer = tc.optim.Adam(model.parameters(),lr=lr)
        for i in  range(st,ed):
            
            train_loss = 0.0
            train_acc = 0.0
            
            x,_ = classifier.preprocess("data/rule_{0}_{1}.trace".format(rule_number,1))
            y = classifier.call_on_batch(x,256)
            
            x,_ = preprocess("data/rule_{0}_{1}.trace".format(rule_number,1))    
            MyDataSet = HeaderData(x,y)
            loader = tc.utils.data.DataLoader(MyDataSet,
                                              batch_size=256,
                                              shuffle=True)
            for batchx,batchy in tqdm(loader):
                batchx,batchy = Variable(batchx),Variable(batchy)
                def closure():
                    nonlocal train_loss,train_acc
                    optimizer.zero_grad()
                    output = model(batchx)
                    loss = loss_fn(output,batchy)
                    train_loss += loss.item()
                    label = torch.max(output, dim=1)[1]
                    train_acc += (label==batchy).sum().item()
                    loss.backward()
                    return loss
                optimizer.step(closure)
            print('file {:d} Train Loss: {:.6f}, Acc: {:.6f}'.format(i,
                  train_loss/x.shape[0],
                  train_acc/x.shape[0]))
    
    train_on_file(1,50,1e-3)
    train_on_file(50,80,1e-4)
    train_on_file(80,100,1e-5)
