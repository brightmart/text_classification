# -*- coding: utf-8 -*-
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
#1.将问题ID和TOPIC对应关系保持到字典里：process question_topic_train_set.txt
#from:question_id,topics(topic_id1,topic_id2,topic_id3,topic_id4,topic_id5)
#  to:(question_id,topic_id1)
#     (question_id,topic_id2)
#read question_topic_train_set.txt
import codecs
#1.################################################################################################################
print("process question_topic_train_set.txt,started...")
q_t='question_topic_train_set.txt'
q_t_file = codecs.open(q_t, 'r', 'utf8')
lines=q_t_file.readlines()
question_topic_dict={}
for i,line in enumerate(lines):
    if i%300000==0:
        print(i)
    #print(line)
    question_id,topic_list_string=line.split('\t')
    #print(question_id)
    #print(topic_list_string)
    topic_list=topic_list_string.replace("\n","").split(",")
    question_topic_dict[question_id]=topic_list
    #for ii,topic in enumerate(topic_list):
    #    print(ii,topic)
    #print("=====================================")
    #if i>10:
    #   print(question_topic_dict)
    #   break
print("process question_topic_train_set.txt,ended...")
###################################################################################################################
###################################################################################################################
#2.处理问题--得到问题ID：问题的表示，存成字典。proces question. for every question form a a list of string to reprensent it.
import codecs
print("process question started11...")
q='question_train_set.txt'
q_file = codecs.open(q, 'r', 'utf8')
q_lines=q_file.readlines()
questionid_words_representation={}
question_representation=[]
length_desc=30
for i,line in enumerate(q_lines):
    #print("line:")
    #print(line)
    element_lists=line.split('\t') #['c324,c39','w305...','c']
    question_id=element_lists[0]
    #print("question_id:",element_lists[0])
    #for i,q_e in enumerate(element_lists):
    #    print("e:",q_e)
    #question_representation=[x for x in element_lists[2].split(",")] #+  #TODO this is only for title's word. no more.
    title_words=[x for x in element_lists[2].strip().split(",")][-length_desc:]
    #print("title_words:",title_words)
    title_c=[x for x in element_lists[1].strip().split(",")][-length_desc:]
    #print("title_c:", title_c)
    desc_words=[x for x in element_lists[4].strip().split(",")][-length_desc:]
    #print("desc_words:", desc_words)
    desc_c=[x for x in element_lists[3].strip().split(",")][-length_desc:]
    #print("desc_c:", desc_c)
    question_representation =title_words+ title_c+desc_words+ desc_c
    question_representation=" ".join(question_representation)
    #print("question_representation:",question_representation)
    #print("question_representation:",question_representation)
    questionid_words_representation[question_id]=question_representation
q_file.close()
print("proces question ended2...")
#####################################################################################################################
###################################################################################################################
# 3.获得模型需要的训练数据。以{问题的表示：TOPIC_ID}的形式的列表
# save training data,testing data: question __label__topic_id
import codecs
import random

print("saving traininig data.started1...")
count = 0
train_zhihu = 'train-zhihu6-title-desc.txt'
test_zhihu = 'test-zhihu6-title-desc.txt'
valid_zhihu = 'valid-zhihu6-title-desc.txt'
data_list = []
multi_label_flag=True

def split_list(listt):
    random.shuffle(listt)
    list_len = len(listt)
    train_len = 0.95
    valid_len = 0.025
    train = listt[0:int(list_len * train_len)]
    valid = listt[int(list_len * train_len):int(list_len * (train_len + valid_len))]
    test = listt[int(list_len * (train_len + valid_len)):]
    return train, valid, test

for question_id, question_representation in questionid_words_representation.items():
    # print("===================>")
    # print('question_id',question_id)
    # print("question_representation:",question_representation)
    # get label_id for this question_id by using:question_topic_dict
    topic_list = question_topic_dict[question_id]
    # print("topic_list:",topic_list)
    # if count>5:
    #    ii=0
    #    ii/0
    if not multi_label_flag:
        for topic_id in topic_list:
            data_list.append((question_representation, topic_id)) #single-label
    else:
        data_list.append((question_representation, topic_list)) #multi-label
    count = count + 1

# random shuffle list
random.shuffle(data_list)

def write_data_to_file_system(file_name, data):
    file = codecs.open(file_name, 'a', 'utf8')
    for d in data:
        # print(d)
        question_representation, topic_id = d
        question_representation_ = " ".join(question_representation)
        file.write(question_representation_ + " __label__" + str(topic_id) + "\n")
    file.close()

def write_data_to_file_system_multilabel(file_name, data):
    file = codecs.open(file_name, 'a', 'utf8')
    for d in data:
        question_representation, topic_id_list = d
        topic_id_list_=" ".join(topic_id_list)
        file.write(question_representation + " __label__" + str(topic_id_list_) + "\n")
    file.close()

train_data, valid_data, test_data = split_list(data_list)
if not multi_label_flag:#single label
    write_data_to_file_system(train_zhihu, train_data)
    write_data_to_file_system(valid_zhihu, valid_data)
    write_data_to_file_system(test_zhihu, test_data)
else:#multi-label
    write_data_to_file_system_multilabel(train_zhihu, train_data)
    write_data_to_file_system_multilabel(valid_zhihu, valid_data)
    write_data_to_file_system_multilabel(test_zhihu, test_data)

print("saving traininig data.ended...")
######################################################################################################################