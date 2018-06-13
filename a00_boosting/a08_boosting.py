# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf

#main process for boosting:
#1.compute label weight after each epoch using validation data.
#2.get weights for each batch during traininig process
#3.compute loss using cross entropy with weights

#1.compute label weight after each epoch using validation data.
def compute_labels_weights(weights_label,logits,labels):
    """
    compute weights for labels in current batch, and update weights_label(a dict)
    :param weights_label:a dict
    :param logit: [None,Vocabulary_size]
    :param label: [None,]
    :return:
    """
    labels_predict=np.argmax(logits,axis=1) # logits:(256,108,754)
    for i in range(len(labels)):
        label=labels[i]
        label_predict=labels_predict[i]
        weight=weights_label.get(label,None)
        if weight==None:
            if label_predict == label:
                weights_label[label]=(1,1)
            else:
                weights_label[label]=(1,0)
        else:
            number=weight[0]
            correct=weight[1]
            number=number+1
            if label_predict==label:
                correct=correct+1
            weights_label[label]=(number,correct)
    return weights_label

#2.get weights for each batch during traininig process
def get_weights_for_current_batch(answer_list,weights_dict):
    """
    get weights for current batch
    :param  answer_list: a numpy array contain labels for a batch
    :param  weights_dict: a dict that contain weights for all labels
    :return: a list. length is label size.
    """
    weights_list_batch=list(np.ones((len(answer_list))))
    answer_list=list(answer_list)
    for i,label in enumerate(answer_list):
        acc=weights_dict[label]
        weights_list_batch[i]=min(1.5,1.0/(acc+0.001))
    #if np.random.choice(200)==0: #print something from time to time
    #    print("weights_list_batch:",weights_list_batch)
    return weights_list_batch

#3.compute loss using cross entropy with weights
def loss(logits,labels,weights):
    loss= tf.losses.sparse_softmax_cross_entropy(labels, logits,weights=weights)
    return loss

#######################################################################
#util function
def get_weights_label_as_standard_dict(weights_label):
    weights_dict = {}
    for k,v in weights_label.items():
        count,correct=v
        weights_dict[k]=float(correct)/float(count)
    return weights_dict
