# -*- coding: utf-8 -*-
import codecs

file='training-data/test-zhihu6-title-desc.txt'
file_x='training-data/test_x.txt'
file_y='training-data/test_y.txt'
file_object = codecs.open(file, 'r', 'utf8')
x_object = codecs.open(file_x, 'a', 'utf8')
y_object = codecs.open(file_y, 'a', 'utf8')

lines=file_object.readlines()
for i,line in enumerate(lines):
    x,y=line.strip().split("__label__")
    #print(i,"x:",x,";y:",y)
    x_object.write(x.strip()+"\n")
    y_object.write(y.strip() + "\n")
    #if i>=10:
    #    break
file_object.close()
x_object.close()
x_object.close()
