# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#最终输出：x1=question_representation,x2=topic_representation,y=0(or 1)--->(x1,x2,y)

import codecs
#1.将问题ID和TOPIC对应关系保持到字典里.################################################################################
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
#2.处理问题--得到{问题ID：问题的表示}，存成字典。
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
    question_representation=[x for x in element_lists[2].split(",")] #+ \
    #[x for x in element_lists[1].split(",")]+ \
    #[x for x in element_lists[4][-length_desc:].split(",")] + \
    #[x for x in element_lists[3][-length_desc*2:].split(",")] #character:字
    #print("question_representation:",question_representation)
    questionid_words_representation[question_id]=question_representation
q_file.close()
print("proces question ended2...")

###################################################################################################################
###################################################################################################################
#3.处理topic，得到{TOPIC_ID,TOPIC的表示}，存成字典
topic_info_file_path='topic_info.txt'
def read_topic_info():
    f = codecs.open(topic_info_file_path, 'r', 'utf8')
    lines=f.readlines()
    dict_questionid_title={}
    for i,line in enumerate(lines):
        topic_id,partent_ids,title_character,title_words,desc_character,decs_words=line.split("\t")
        # print(i,"------------------------------------------------------")
        # print("topic_id:",topic_id)
        # print("partent_ids:",partent_ids)
        # print("title_character:",title_character)
        # print("title_words:",title_words)
        # print("desc_character:",desc_character)
        # print("decs_words:",decs_words)
        decs_words=decs_words.strip()
        decs_words = decs_words.strip().split(",");decs_words = " ".join(decs_words)

        title_words=title_words.strip().split(",");title_words=title_words[0:30];title_words=" ".join(title_words);
        dict_questionid_title[topic_id]=title_words+" "+decs_words
    print("len(dict_questionid_title):",len(dict_questionid_title))
    return dict_questionid_title
dict_questionid_title=read_topic_info()
#####################################################################################################################
#####################################################################################################################
# 4.获得模型需要的训练数据。以{问题的表示：TOPIC_ID}的形式的列表
# save training data,testing data: question __label__topic_id
import codecs
import random

print("saving traininig data.started1...")
count = 0
train_zhihu = 'train_twoCNN_zhihu.txt'
test_zhihu =  'test_twoCNN_zhihu.txt'
valid_zhihu = 'valid_twoCNN_zhihu.txt'
data_list = []


def split_list(listt):
    random.shuffle(listt)
    list_len = len(listt)
    train_len = 0.9
    valid_len = 0.05
    train = listt[0:int(list_len * train_len)]
    valid = listt[int(list_len * train_len):int(list_len * (train_len + valid_len))]
    test = listt[int(list_len * (train_len + valid_len)):]
    return train, valid, test

topic_list_previous=None
from random import choice
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
    for topic_id in topic_list:
        topic_representation=dict_questionid_title[topic_id]
        data_list.append((question_representation, topic_representation,1))
        if topic_list_previous is not None:
            topic_representation_negative = dict_questionid_title[choice(topic_list_previous)]
            data_list.append((question_representation,topic_representation_negative,0))
    topic_list_previous=topic_list
    count = count + 1

# random shuffle list
random.shuffle(data_list)


def write_data_to_file_system(file_name, data):
    file = codecs.open(file_name, 'a', 'utf8')
    for d in data:
        # print(d)
        question_representation,topic_representation, label = d
        question_representation_ = " ".join(question_representation)
        file.write(question_representation_ + "\t"+topic_representation+" __label__" + str(label) + "\n")
    file.close()


train_data, valid_data, test_data = split_list(data_list)
write_data_to_file_system(train_zhihu, train_data)
write_data_to_file_system(valid_zhihu, valid_data)
write_data_to_file_system(test_zhihu, test_data)
print("saving traininig data.ended...")
######################################################################################################################