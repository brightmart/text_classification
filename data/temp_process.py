# -*- coding: utf-8 -*-
import codecs

def read_split_write(file_path,target_file_path):
    #1.read data
    file_object=codecs.open(file_path,mode='r',encoding='utf-8')
    target_object=codecs.open(target_file_path,mode='a',encoding='utf-8')
    lines=file_object.readlines()
    print("length of lines:",len(lines))
    for i,line in enumerate(lines):
        #2.split data.
        input_string,labels=line.strip().split("__label__")
        label_list=labels.split(" ")
        #3.write data
        target_object.write(input_string.strip()+" ")
        for label in label_list:
            target_object.write("__label__"+str(label)+" ")
        target_object.write("\n")
    target_object.close()
    file_object.close()

file_path='sample_multiple_label.txt'
target_file_path='sample_multiple_label3.txt'
read_split_write(file_path,target_file_path)
