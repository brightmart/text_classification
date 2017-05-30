# text_classification
the purpose of this repository is explore text classification methods in NLP with deep learning. 
it has all kinds of baseline models for text classificaiton. although many of these models are simple, and may not get you to top level of the tasks.but some of these models are very classic, so they may be good to serve as baseline models.

Useage:

1) model is in xxx_model.py
2) run python xxx_train.py to train the model
3) run python xxx_predict.py to do inference(test).

Notice:

some util function is in data_util.py; 
typical input like: "x1 x2 x3 x4 x5 __label__ 323434" where 'x1,x2' is words, '323434' is label;
it has a function to load and assign pretrained word embedding to the model,where word embedding is pretrained in word2vec or fastText. 

Models:

1.fastText:  implmentation of <a href="https://arxiv.org/abs/1607.01759">Bag of Tricks for Efficient Text Classification</a>
1) use bi-gram and/or tri-gram
2) use NCE loss to speed us softmax computation(not use hierarchy softmax as original paper)
result: performance is as good as paper, speed also very fast.
check: p5_fastTextB_model.py

2.TextCNN: implementation of <a href="http://www.aclweb.org/anthology/D14-1181"> Convolutional Neural Networks for Sentence Classification </a>

structure:embedding--->conv--->max pooling--->fully connected layer-------->softmax
check: p7_TextCNN_model.py

3.TextRNN
structure:embedding--->bi-directional lstm--->concat output--->average----->softmax
check: p8_TextRNN_model.py

4.BiLstmTextRelation
structure same as TextRNN. but input is special designed. e.g.input:"how much is the computer? EOS price of laptop". where 'EOS' is a special
token spilted question1 and question2.
check:p9_BiLstmTextRelation_model.py

5.twoCNNTextRelation
structure: first use two different convolutional to extract feature of two sentences. then concat two features. use linear
transform layer to out projection to target label, then softmax.
check: p9_twoCNNTextRelation_model.py

6.BiLstmTextRelationTwoRNN
structure: one bi-directional lstm for one sentence(get output1), another bi-directional lstm for another sentence(get output2). then:
softmax(output1*M*output2)
check:p9_BiLstmTextRelationTwoRNN_model.py
for more detail you can go to: <a herf='http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/'>Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow<a>

to be continued. for any problem, concat brightmart@hotmail.com
