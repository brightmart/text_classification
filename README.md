Text Classification
the purpose of this repository is explore text classification methods in NLP with deep learning. 

it has all kinds of baseline models for text classificaiton.

it also support for multi-label classification where multi label associate with an sentence or document.

although many of these models are simple, and may not get you to top level of the task.but some of these models are very classic, so they may be good to serve as baseline models.

Model including:

1)fastText 2)TextCNN 3)TextRNN 4)BiLstmTextRelation 5)twoCNNTextRelation 6)BiLstmTextRelationTwoRNN 

7)RCNN 8)Hierarchical Attention Network 9)seq2seq with attention

-------------------------------------------------------------------------

Performance
(mulit-label label prediction task,ask to prediction top5, 3 million training data,full mark:0.5)

Model    | fastText  | TextCNN | TextRNN | RCNN  | Hierarchical Attention Network
---      | ---       | ---     | ---     |---    |---                            |
Score    | 0.362     |  0.405  |  0.358  | 0.395 | 0.398 
Training | 10 minutes| 2 hours | 10 hours|2 hours| 2 hours
-------------------------------------------------------------------------

Useage:

1) model is in xxx_model.py
2) run python xxx_train.py to train the model
3) run python xxx_predict.py to do inference(test).

Each model has a test method under the model class. you can run the test method first to check whether the model can work properly.

-------------------------------------------------------------------------

Notice:

Some util function is in data_util.py; 
typical input like: "x1 x2 x3 x4 x5 __label__ 323434" where 'x1,x2' is words, '323434' is label;
it has a function to load and assign pretrained word embedding to the model,where word embedding is pretrained in word2vec or fastText. 

Models:

-------------------------------------------------------------------------

1.fastText:  implmentation of <a href="https://arxiv.org/abs/1607.01759">Bag of Tricks for Efficient Text Classification</a>
1) use bi-gram and/or tri-gram
2) use NCE loss to speed us softmax computation(not use hierarchy softmax as original paper)
result: performance is as good as paper, speed also very fast.

check: p5_fastTextB_model.py

-------------------------------------------------------------------------

2.TextCNN: implementation of <a href="http://www.aclweb.org/anthology/D14-1181"> Convolutional Neural Networks for Sentence Classification </a>

Structure:embedding--->conv--->max pooling--->fully connected layer-------->softmax

check: p7_TextCNN_model.py

in order to get very good result with TextCNN, you also need to read carefully about this paper <a href="https://arxiv.org/abs/1510.03820">A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification</a>: it give you some insights of things that can affect performance. although you need to  change some settings according to your specific task.


-------------------------------------------------------------------------


3.TextRNN
Structure:embedding--->bi-directional lstm--->concat output--->average----->softmax

check: p8_TextRNN_model.py


-------------------------------------------------------------------------


4.BiLstmTextRelation
Structure same as TextRNN. but input is special designed. e.g.input:"how much is the computer? EOS price of laptop". where 'EOS' is a special
token spilted question1 and question2.

check:p9_BiLstmTextRelation_model.py


-------------------------------------------------------------------------


5.twoCNNTextRelation
Structure: first use two different convolutional to extract feature of two sentences. then concat two features. use linear
transform layer to out projection to target label, then softmax.

check: p9_twoCNNTextRelation_model.py


-------------------------------------------------------------------------


6.BiLstmTextRelationTwoRNN
Structure: one bi-directional lstm for one sentence(get output1), another bi-directional lstm for another sentence(get output2). then:
softmax(output1*M*output2)

check:p9_BiLstmTextRelationTwoRNN_model.py

for more detail you can go to: <a herf="http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow">Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow<a>


-------------------------------------------------------------------------


7.RCNN:
Recurrent convolutional neural network for text classification

implementation of <a href="https://scholar.google.com.hk/scholar?q=Recurrent+Convolutional+Neural+Networks+for+Text+Classification&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0ahUKEwjpx82cvqTUAhWHspQKHUbDBDYQgQMIITAA"> Recurrent Convolutional Neural Network for Text Classification </a>
 
structure:1)recurrent structure (convolutional layer) 2)max pooling 3) fully connected layer+softmax

it learn represenation of each word in the sentence or document with left side context and right side context:

representation current word=[left_side_context_vector,current_word_embedding,right_side_context_vecotor].

for left side context, it use a recurrent structure, a no-linearity transfrom of previous word and left side previous context; similarly to right side context.

check: p71_TextRCNN_model.py


-------------------------------------------------------------------------

8.Hierarchical Attention Network:

Implementation of <a href="https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf">Hierarchical Attention Networks for Document Classification</a>

Structure:

1)embedding 

2) Word Encoder: word level bi-directional GRU to get rich representation of words

3) Word Attention:word level attention to get important information in a sentence

4) Sentence Encoder: sentence level bi-directional GRU to get rich representation of sentences

5) Sentence Attetion: sentence level attention to get important sentence among sentences

5) FC+Softmax


Input of data: 

Generally speaking, input of this model should have serveral sentences instead of sinle sentence. shape is:[None,sentence_lenght]. where None means the batch_size.

In my training data, for each example, i have four parts. each part has same length. i concat four parts to form one single sentence. the model will split the sentence into four parts, to form a tensor with shape:[None,num_sentence,sentence_length]. where num_sentence is number of sentences(equal to 4, in my setting).

check:p1_HierarchicalAttention_model.py

-------------------------------------------------------------------------

9.Seq2seq with attention

Implementation seq2seq with attention derived from <a href="https://arxiv.org/pdf/1409.0473.pdf">NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE</a>

I.Structure:

1)embedding 2)bi-GRU too get rich representation from source sentences(forward & backward). 3)decoder with attention.

II.Input of data:

there are two kinds of three kinds of inputs:1)encoder inputs, which is a sentence; 2)decoder inputs, it is labels list with fixed length;3)target labels, it is also a list of labels.

for example, labels is:"L1 L2 L3 L4", then decoder inputs will be:[_GO,L1,L2,L2,L3,_PAD]; target label will be:[L1,L2,L3,L3,_END,_PAD]. length is fixed to 6, any exceed labels will be trancated, will pad if label is not enough to fill.

III.Attention Mechanism:

1) transfer encoder input list and hidden state of decoder

2) calculate similiarity of hidden state with each encoder input, to get possibility distribution for each encoder input.

3) weighted sum of encoder input based on possibility distribution.

   go though RNN Cell using this weight sum together with decoder input to get new hidden state

IV.How Vanilla Encoder Decoder Works:

the source sentence will be encoded using RNN as fixed size vector ("thought vector"). then during decoder:

1) when it is training, another RNN will be used to try to get a word by using this "thought vector"  as init state, and take input from decoder input at each timestamp. decoder start from special token "_GO". 
after one step is performanced, new hidden state will be get and together with new input, we can continue this process until we reach to a special token "_END". 
we can calculate loss by compute cross entropy loss of logits and target label. logits is get through a projection layer for the hidden state(for output of decoder step(in GRU we can just use hidden states from decoder as output).

2) when it is testing, there is no label. so we should feed the output we get from previous timestamp, and continue the process util we reached "_END" TOKEN.

V.Notices:

1) here i use two kinds of vocabularies. one is from words,used by encoder; another is for labels,used by decoder

2) for vocabulary of lables, i insert three special token:"_GO","_END","_PAD"; "_UNK" is not used, since all labels is pre-defined.

-------------------------------------------------------------------------

TODO 

1.Character-level Convolutional Networks for Text Classification

2.Convolutional Neural Networks for Text Categorization:Shallow Word-level vs. Deep Character-level

3.Very Deep Convolutional Networks for Text Classification

4.Memory network


-------------------------------------------------------------------------

Reference:

1.Bag of Tricks for Efficient Text Classification

2.Convolutional Neural Networks for Sentence Classification

3.Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow, from www.wildml.com

4.Recurrent Convolutional Neural Network for Text Classification

5.Hierarchical Attention Networks for Document Classification

-------------------------------------------------------------------------

to be continued. for any problem, concat brightmart@hotmail.com
