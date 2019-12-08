

import tensorflow as tf
from tensorflow import keras
import numpy as np

rule_number = 1000

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
    head = tf.convert_to_tensor(head)
    
#    head = tf.one_hot(head,256)
#    head = tf.reshape(head,(-1,256*13))
#    head = tf.cast(head,tf.float32)
    
    head = head/128 - 1
    index = tf.convert_to_tensor(A[:,6])
    index = tf.one_hot(index,rule_number)
    #train_db = tf.data.Dataset.from_tensor_slices((head,index))
    
    return head,index

class classifier(keras.Model):
    def __init__(self):
        super(classifier,self).__init__()
        #self.fc1 = keras.layers.Dense(64,activation="relu")
        #self.fc1 = keras.layers.Dense(1024,activation="relu")
        self.fc0 = keras.layers.Dense(256,activation="relu")
        self.fc1 = keras.layers.Dense(256,activation="relu")
        self.fc2 = keras.layers.Dense(256,activation="relu")
        self.fc3 = keras.layers.Dense(rule_number,activation="softmax")
        
        
    def call(self,x,training=None):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = classifier()
    #model.build((None,13))
    model.compile(optimizer=tf.optimizers.RMSprop(1e-4),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
    for i in  range(1,101):
        x,y = preprocess("data/rule_{0}_{1}.trace".format(rule_number,i))        
        model.fit(x,y,batch_size=256,epochs=1)


for i in  range(1,101):
    x,y = preprocess("data/rule_{0}_{1}.trace".format(rule_number,i))
    model.call(x)

