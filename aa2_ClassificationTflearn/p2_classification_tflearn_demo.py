print("started...")
import tflearn
import numpy as np
import tensorflow as tf
class_number=3 #10
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 784]) #set input data's shape
net = tflearn.fully_connected(net, 64) #one layer of FC
net = tflearn.dropout(net, 0.5) #dropout
net = tflearn.fully_connected(net, class_number, activation='softmax') #one layer of FC(this layer will be output possibility distribution )
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy') #classification

model = tflearn.DNN(net) #deep Neural Network Model

# invoke method with some data--------------------------------------------------------------------
def convert_int_to_one_hot(number,label_size):
    listt=[0 for x in range(label_size)]
    listt[number]=1
    return listt

batch_size=32
X=np.random.randn(batch_size,784)
y_=[convert_int_to_one_hot(np.random.choice(class_number),class_number) for xx in range(batch_size)]
y=np.array(y_)

X_test=np.random.randn(batch_size,784)
y_2=[convert_int_to_one_hot(np.random.choice(class_number),class_number) for xx in range(batch_size)]
Y_test=np.array(y_2)
model.fit(X, y,validation_set=(X_test, Y_test),show_metric=True)
#-------------------------------------------------------------------------------------------------
print("ended...")