

Project Report

Question Answering Network and analysis of different techniques.

Sunny Shukla shukla.su@northeastern.edu 001313981

Kartik Sharma sharma.kart@northeastern.edu 001087313

Gaurav Singh singh.gau@northeastern.edu 001063386

Manjit Ullal ullal.m@northeastern.edu 001374628

Khoury College of Computer Sciences

Northeastern University

Boston, MA 02115

Question Answering or reading comprehension is the

The structure of this report is as follows. We ﬁrst begin

by introducing the dataset and do some exploratory analy-

sis on the dataset. We then introduce the models and show

our experiments. Finally, we look at the learnings and future

work.

ability to read a given text and then answer questions

about it. It is an important task based on which in-

telligence of NLP systems and AI in general can be

judged. This, is a challenging task for machines, requir-

ing both understandings of natural language and knowl-

edge about the world. The system must select the an-

swer from all possible spans in the passage, thus need-

ing to cope with a fairly large number of candidates.

Dataset

The Stanford Question Answering Dataset (SQuAD 1.1),

is a reading comprehension dataset consisting of 100,000+

questions posed by crowdworkers on a set of Wikipedia arti-

cles, where the answer to each question is a segment of text

from the corresponding reading passage.

The Wikipedia articles were selected from project

Nayuki’s Wikipedia’s internal PageRanks to obtain the top

10000 articles of English Wikipedia, from which 536 arti-

cles were sampled uniformly at random. Paragraphs with

less than 500 characters, tables, and images were discarded.

The result was 23,215 paragraphs for the 536 articles cov-

ering a wide range of topics, from musical celebrities to ab-

stract concepts.

On each paragraph, crowdworkers were tasked with ask-

ing and answering up to 5 questions on the content of that

paragraph Among the answers the shortest span in the para-

graph that answered the question was selected. On average

4.8 answers were collected per question.

A major issue with SQuAD 1.1 was that the models only

needed to select the span that seemed most related to the

question, instead of checking that the answer is actually en-

tailed by the text. To ﬁx the issue the SQuAD 2.0 dataset

was released which combines existIng SQuAD data with

over 50,000 unanswerable questions written adversarially by

crowdworkers to look similar to answerable ones. This en-

sures that models must not only answer questions when pos-

sible but also determine when no answer is supported by the

paragraph and abstain from answering.

In this project we attempt at implementing some of

the most important papers for Question Answering and

evaluate their performance. We compare different tech-

niques for QA including training models from end to

end as well as ﬁne tuning larger language models. To

evaluate the performance of our models we used the

SQuAD 2.0 dataset (1).

Introduction

The tasks of machine comprehension (MC) and question

answering (QA) are becoming an essential benchmark for

model evaluation in recent years. This is an important NLP

task that involves extracting an answer from a given passage.

The bigger problem of general question answering has many

potential applications including building intelligent systems.

While not exactly aiming for this level of complexity, in this

project we are aiming to measure how well a machine can

answer general questions by extraction, based on paragraphs

from Wikipedia articles. Despite the seemingly easy nature

from a human point of view, it remains to be a complex chal-

lenge for machines.

With the rise of deep learning-based models and more

and more public available datasets, we have seen signiﬁ-

cant progress on this task. In this project, we look at four

different approaches to solving the QA challenges. As the

attention mechanism has been indispensable in recent suc-

cessful models. In this project, we investigate two such

attention-based models, BiDAF and QANet. We also look

at the more recent larger language models and investigate

two such ﬂavors of pre-trained language models ALBERT

and RoBERTa.

The dataset contains 130,319 questions in the Training set

and 11,873 questions in the Validation set along with 8,862

questions in the Test set. In our project, we will be using the

train and validation splits of the SQuAD 2.0 dataset (1). The

SQuAD dataset mentions the following metrics for evalua-

tion and we will be evaluating performance on both of them.

Copyright © 2020, Association for the Advancement of Artiﬁcial

Intelligence (www.aaai.org). All rights reserved.





Exact match

This metric measures the percentage of predictions that

match any one of the ground truth answers exactly.

F1 score (Macro-averaged)

This metric measures the average overlap between the pre-

diction and ground truth answer. The prediction and ground

truth tokens are treated as bags of tokens and compute their

F1. The maximum F1 over all of the ground truth answers

for a given question is taken, and then averaged over all of

the questions.

After this, we tried to plot the question types for all these

categories of title types and came up with the following

plot. It shows that most of the questions for each label type

ask about persons, organizations, places, dates, and num-

bers which do not fall under other categories. Some relation-

ships seen were: Norp(Nationality, religious groups) links

with GPE(places, countries, cities, etc.) and dates. Organi-

zations link with Person, organization, places, dates, ranks,

and other kinds of numbers.

Analysis

In the SQuAD2.0 dataset, we have tried to explore 75 per-

cent of the training data. We wanted to come up with the

answer to “type” of questions, answers, and context along-

side numbers. Upon further analysis, we came up with con-

cepts of LDA (Latent Dirichlet Allocation) but LDA re-

quired training which would mean we need to have labeled

data. We have unlabeled data and now this could be a project

in itself to cluster unlabeled datasets. We also came across

different methods to deal with the problem of unlabeled data,

some of them were BERT-based models that could deal with

semi unlabeled data but going that way would mean de-

viation from our current project title. Just for data explo-

ration, we required something pretrained and generalized

enough in a way that could be just like having an optimal

N number of clusters of unlabeled data. Spacy name entity

recognition seemed a perfect choice for this task as it’s pre-

trained and seems sophisticated enough for the task of data

exploration. We used Spacy’s “en core web sm” library as

its smaller than other two available and yet has similar F1

rates. It has 18 types of data categories namely ‘PERSON’,

’NORP’, ’FAC’, ’ORG’, ’GPE’, ’LOC’, ’PRODUCT’,

’EVENT’, ’WORK OF ART’, ’LAW’, ’LANGUAGE’,

’DATE’, ’TIME’, ’PERCENT’, ’MONEY’, ’QUANTITY’,

’ORDINAL’, ’CARDINAL’. For the task of type of ques-

tion asked, we passed context, title, and combined question-

answer to Spacy NER algorithm. We found that Spacy

doesn’t work with single entities i.e. if we pass just a per-

son’s name it returns None, as it is based upon the grammar

of the language and just one word could mean a noun, ad-

jective, or anything. We then generated title-labels from the

context labels if the context contained the title within. That’s

how we came up with title labels. Then we passed combined

question and answers hoping that this way it can return much

better labels for question-answer and we were successful.

Next, we tried to ﬁnd the number of each label type in titles

and for each of those titles, the number of questions type.

We found from the following plot that the majority of titles

were of three types i.e. Person, organization, GPE (coun-

tries, states, cities).

After this, we tried to analyze our data from the length

of context, questions and answers perspective and we found

the following ﬁgures suggesting that most answers are of

length 6, while most contexts are of length 500-600 charac-

ters and most questions have a length of 110 approximately.

One point to be noted is that we tried to normalize this data

by choosing distinct values of all contexts, questions, and

answers.





features in an image by operating on its pixels, here they’ll

do so by operating on characters of words.

Highway networks were originally introduced to ease the

training of deep neural networks. The purpose of this layer is

to learn to pass relevant information from the input. A high-

way network is a series of feed-forward or linear layers with

a gating mechanism. The gating is implemented by using a

sigmoid function which decides what amount of informa-

tion should be transformed and what should be passed as it

is. The input to this layer is the concatenation of word and

character embeddings of each word. The idea here is that the

adding of highway layers enables the network to make more

efﬁcient use of character embeddings. If a particular word

is not found in the pretrained word vector vocabulary (OOV

word), it will most likely be initialized with a zero vector.

It then makes much more sense to look at the character em-

bedding of that word rather than the word embedding. The

soft gating mechanism in highway layers helps the model to

achieve this.

Attention Flow Layer is responsible for fusing and link-

ing the context and query representations. This layer cal-

culates attention in two directions: from context to query

and from query to context. Attention vectors for these cal-

culations are derived from a common matrix which is called

as the similarity matrix. Modeling Layer is responsible for

capturing temporal features interactions among the context

words. This is done using a bidirectional LSTM. The dif-

ference between this layer and the contextual layer, both of

which involve an LSTM layer is that here we have a query-

aware representation of the context while in the contextual

layer, encoding of the context and query was independent.

Contextual Embedding is the ﬁnal embedding layer in the

model. The output of highway layers is passed to a bidirec-

tion LSTM to model the temporal features of the text. This

is done for both, the context and the query.

Models

BiDAF

The paper (3) implements is a multi-stage hierarchical ar-

chitecture that represents the context and query at multiple

levels of granularity. It also involves recurrence as it exten-

sively uses LSTMs and a memory-less attention mechanism

which is bi-directional in nature. Below are some important

points regarding the BiDAF model.

The diagram of the BiDAF model is shown below.

The key issue that this paper tries to address is that of

early summarization in all the earlier approches that use at-

tention mechanisms. The attention mechanisms until then

were used to obtain a ﬁxed-size summarization of given val-

ues and query. This, according to the authors leads to early

summarization and loss of information. Moreover, previ-

ously, attention was only calculated in only one direction.

To improve upon these issues, the authors propose a hierar-

chical, multi-stage network.

Word embedding layers maps each word to a high-

dimensional vector space. We use pre-trained word vectors,

GloVe to obtain the ﬁxed word embedding of each word.

A character embedding is calculated for each context and

query word. This is done by using convolutions. It maps each

word to a vector space using character-level CNNs. Here,

we trained a simple CNN with one layer of convolution on

top of pretrained word vectors and hypothesized that these

pretrained word vectors could work as a universal feature

extractors for various classiﬁcation tasks. This is analogous

to the earlier layers of vision models like VGG and Incep-

tion working as generic feature extractors. The intuition is

simple over here. Just as convolutional ﬁlters learn various

Running BiDAF for 10 epochs we get F1 score equal to

56.46 and EM equal to 68.32.

QANet

The paper (2) draws inspiration from "Attention Is All You

Need". The key motivation behind the design of the model

is: convolution captures the local structure of the text, while

the self-attention learns the global interaction between each

pair of words. Below are some important points regarding

the QANet model.





Other papers have been heavily based on recurrent neu-

ral nets and attention. However, RNNs are slow to train

given their sequential nature and are also slow for inference.

QANet was proposed in early 2018. This paper does away

with recurrence and is only based on self-attention and con-

volutions.

Depthwise separable convolutions serve the same purpose

as normal convolutions with the only difference being that

they are faster because they reduce the number of multipli-

cation operations. This is done by breaking the convolution

operation into two parts: depthwise convolution and point-

wise convolution. Depthwise convolutions are faster than

traditional convolution as number of computations in depth-

wise separable convolutions are lesser than traditional ones.

Highway network used here is same as that used in BiDAF

model.

Embedding Layer converts ord-level tokens into a 300-

dim pre-trained glove embedding vector, creates trainable

character embeddings using 2-D convolutions and concate-

nates character and word embeddings and passes them

through a highway network. Self Attention is same as dis-

cussed in the BiDAF model. The attention layer is the core

building block of the network where the fusion between con-

text and query occurs. QANet uses trilinear attention func-

tion used in BiDAF model.

Running QANet for 10 epochs we get F1 score equal to

73.41 and EM equal to 62.52.

BERT

The ﬁeld of NLP has been transformed with the recent ad-

vent of pre-trained language models. These language models

are ﬁrst trained on a large corpus of text and then are ﬁne-

tuned on a downstream task. BERT (Bidirectional Encoder

Representations from Transformers) (4) is one such model

that uses employs a stacked transformer-based architecture

and bidirectional training.

For Encoder Layer a positional embedding is injected into

the input. This is then passed through a series of convolu-

tional layers. The number of these layers depend upon the

layer of which these encoder blocks are a part of. The out-

put of this is then passed to a multiheaded self attention layer

and ﬁnally to a feedforward network which is simply a linear

layer. The model also involves residual connections, layer

normalizations and dropouts. The encoder layer is shown as

below.

BERT’s model architecture is a multi-layer bidirectional

Transformer with only the encoder part. To make BERT

handle a variety of down-stream tasks, the input represen-

tation can unambiguously represent both a single sentence

and a pair of sentences in one token sequence. To represent a

word/token it uses WordPiece embeddings (7) with a 30,000

token vocabulary. The ﬁrst token of every sequence is al-

ways a special classiﬁcation token ([CLS]). The ﬁnal hidden

state corresponding to this token is used as the aggregate se-

quence representation for classiﬁcation tasks. Sentence pairs

are packed together into a single sequence. Sentence dif-

ferentiation is done in two ways ﬁrstly by separating them

with a special token ([SEP]). Additionally, a learned embed-

ding is added to every token indicating whether it belongs

to sentence A or sentence B For a given token, its input

representation is constructed by summing the correspond-

ing token, segment, and position embeddings It is trained on

a large corpus with two objectives Masked language model-

ing (MSM ) and Next Sentence Prediction (NSP). Although

BERT has given SOTA results on many NLP tasks it is still

cumbersome to train and ﬁne-tune due to a large number

of parameters. As such it takes a long time to train and the

training is compute intensive.

ALBERT

ALBERT (A Lite BERT) (5) is an update on BERT that ad-

dresses these problems, utilizing parameter reduction. The

parameter reduction techniques allow for the models to

The diagram of an QANet model is show below.





have a lesser number of parameters and make it easier to

train and require lesser compute. The parameter reduction

is done using the following approaches. Factorization of

the embedding parametrization — Input-level embeddings

(words, sub-tokens, etc.) need to learn context-independent

representations. To do so, the embedding matrix is split

between input-level embeddings with a relatively-low di-

mension (e.g., 128), while the hidden-layer embeddings use

higher dimensionalities (768 as in the BERT case, or more).

This allows for about 80% reduction in the number of pa-

rameters. Parameter-sharing across the layers - Transformer-

based neural network architectures rely on independent lay-

ers stacked on top of each other. But the network often learns

to perform similar operations at various layers, using differ-

ent parameters of the network. In ALBERT parameters are

shared across the layers, i.e., the same layer is applied on

top of each other. Implementing these two design changes

together yields an ALBERT-base model that has only 12M

parameters, an 89% parameter reduction compared to the

BERT-base model. We have employed the huggingface’ AL-

BERT base model and ﬁne-tuned it on the SQUAD (1)

dataset.

Evaluation

We used F1 score and EM score for evaluation of differ-

ent models. However, all models have different architecture

as per represented in their respective papers (references at

the end of the report). Langauage models take time to train

hence we used Collab PRO for each experiment. BiDAF,

QANet and ALBERT models took almost 10 hours to run for

10 epochs. For RoBERTa model, it took 6 hours to run for 3

epochs. The result shown in this report isn’t anywhere close

to desired results, but given a server with high power GPU

which can handle expensive computation the models which

we design by hand or ﬁne tuned can be trained to reached

the performace as described in their respective papers.

The ALBERT model was ﬁne tuned using AdamW as the

optimizer and the learning rate as 5 × 10−5. The batch size

was 8 and the model was trained for 9 epochs. The RoBERTa

model was also ﬁne tuned using AdamW as the optimizer

and the learning rate as 5 × 10−5. The batch size was 8, and

it was trained for 3 epochs. For QANet we use a learning

rate warm-up scheme with an inverse exponential increase

from 0.0 to 0.001 in the ﬁrst 1000 steps, and then maintain a

constant learning rate for the remainder of training. We used

ADAM as the optimizer and it was trained for 10 epochs. For

BiDAF we use the AdaDelta optimizer, with a mini batch

size of 60 and an initial learning rate of 0.5. It was trained

for 10 epochs.

BiDAF: F1 = 56.46, EM = 68.32

ROBERTA

RoBERTa: (A Robustly Optimized BERT Pretraining Ap-

proach)(6) builds on BERT’s language masking strategy,

and is retraining of BERT with improved training methodol-

ogy, more data, and compute power. Essentially it makes the

following changes to the original BERT implementation. It

uses more data: 160GB of text instead of the 16GB dataset

originally used to train BERT. It also trains for a longer pe-

riod: increasing the number of iterations from 100K to 300K

and then further to 500K. RoBERTa is also trained using

larger batches: 8K instead of 256 in the original BERT base

model, and also employs byte-level BPE vocabulary with

50K subword units instead of character-level BPE vocabu-

lary of size 30K. Finally, it removes the next sequence pre-

diction objective from the training procedure and dynam-

ically changes the masking pattern applied to the training

data. We have employed the huggingface’ RoBERTa base

model and ﬁne-tuned it on the SQUAD(1) dataset.





QANet: F1 = 73.41, EM = 62.52

RoBERTa: F1 = 83.83, EM = 74.93

ALBERT: F1 = 84.36, EM = 74.52





Conclusion

ing. Some of the interesting learnings we observed were as

follows.

In this project, we implemented QANET and BiDAF and

ﬁne-tuned ALBERT and RoBERTa models on the SQuAD

2.0 dataset (1). The results we obtained on the dev set are

displayed in the below table. From our experiments, we ob-

serve that ALBERT performed the best on the F1 score met-

ric with an average score of 84.36, Whereas RoBERTa per-

formed marginally better on the exact match metric with

a value of 74.93 ( although due to compute constraints it

was trained for lesser epochs). Our empirical results come

in line with the expected results. The pretrained language

models outperformed the end to end trained models by a

huge margin, achieving better results in the ﬁrst epoch it-

self. These results show the beneﬁt these large models enjoy

due to a large number of parameters and the large dataset

they are trained on. Another interesting observation we ob-

serve is that the larger models almost plateaued after a cou-

ple of epochs and no major improvement was observed in

later epochs (in fact the models performed slightly poor in

the later epochs). Whereas, QANet continued to improve

with subsequent epochs. In terms of non-pretrained models,

QANet performed better than BiDAF.

• Data preprocessing and modeling the data in a shape that

the model understands is a much more challenging task

than creating the model itself.

• The attention mechanism and the huge transformative im-

pact it has had on NLP. It was a challenging yet rewarding

process of understanding and making it work in accor-

dance with different architectures.

• The major advancement in NLP in terms of Pre-trained

language models. Discounting the compute and training

time, these models outperformed the other models from

the get-go. It also made us aware of the huge impact li-

braries like huggingface have in making these models ac-

cessible to the public.

• The importance of hyperparameter tuning, especially

when training larger models. Batch size and learning rate

played a huge role in determining the ease of training

these models.

Future Work

Model

BiDAF

Epochs

F1

EM

Due to time and resource constraints, we were unable to

perform some experiments and would be part of our fu-

ture work. It would be an interesting exercise to see how

much improvement can be achieved by ensembling the re-

sults from all the models. Another interesting experiment

would be to take the trained/ﬁnetuned models and see how

well the results transfer to other question answering datasets

like Natural Questions (NQ), Question Answering in Con-

text (QuAC), Conversational Question Answering(CoQA),

etc.

10

56.46 68.32

(Hidden Dim: 100)

QANET

10

73.41 62.52

(Hidden Dim:128,

Attention Head: 8)

ALBERT

9

3

84.36 74.52

83.83 74.93

RoBERTa

Contribution

MEMBERS

TASKS

Sunny Shukla

QANet, BiDAF and Project Report

Kartik Sharma ALBERT, RoBERTa and Project Report

Gaurav Singh

Manjit Ullal

SQUAD, RoBERTa and Project Report

SQUAD, BiDAF and Project Report

References

[1.](https://rajpurkar.github.io/SQuAD-explorer/)[ ](https://rajpurkar.github.io/SQuAD-explorer/)[SQUAD](https://rajpurkar.github.io/SQuAD-explorer/)[ ](https://rajpurkar.github.io/SQuAD-explorer/)[Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

[2.](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[QANet:](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Combining](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Local](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Convolution](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[with](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Global](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Self-](https://arxiv.org/abs/1804.09541)

[Attention](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[for](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Reading](https://arxiv.org/abs/1804.09541)[ ](https://arxiv.org/abs/1804.09541)[Comprehension](https://arxiv.org/abs/1804.09541)

[3.](https://arxiv.org/abs/1611.01603)[ ](https://arxiv.org/abs/1611.01603)[Bidirectional](https://arxiv.org/abs/1611.01603)[ ](https://arxiv.org/abs/1611.01603)[Attention](https://arxiv.org/abs/1611.01603)[ ](https://arxiv.org/abs/1611.01603)[Flow](https://arxiv.org/abs/1611.01603)[ ](https://arxiv.org/abs/1611.01603)[for](https://arxiv.org/abs/1611.01603)[ ](https://arxiv.org/abs/1611.01603)[Machine](https://arxiv.org/abs/1611.01603)[ ](https://arxiv.org/abs/1611.01603)[Comprehension](https://arxiv.org/abs/1611.01603)

[4.](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[BERT:](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[Pre-training](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[of](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[Deep](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[Bidirectional](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[Transformers](https://arxiv.org/pdf/1810.04805.pdf)

[for](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[Language](https://arxiv.org/pdf/1810.04805.pdf)[ ](https://arxiv.org/pdf/1810.04805.pdf)[Understanding](https://arxiv.org/pdf/1810.04805.pdf)

[5.](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[ALBERT:](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[A](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[LITE](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[BERT](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[FOR](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[SELF-SUPERVISED](https://arxiv.org/pdf/1909.11942.pdf)

[LEARNING](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[OF](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[LANGUAGE](https://arxiv.org/pdf/1909.11942.pdf)[ ](https://arxiv.org/pdf/1909.11942.pdf)[REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf)

[6.](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[RoBERTa:](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[A](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[Robustly](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[Optimized](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[BERT](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[Pretraining](https://arxiv.org/pdf/1907.11692.pdf)[ ](https://arxiv.org/pdf/1907.11692.pdf)[Ap-](https://arxiv.org/pdf/1907.11692.pdf)

[proach](https://arxiv.org/pdf/1907.11692.pdf)

[7.](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Google’s](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Neural](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Machine](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Translation](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[System:](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Bridging](https://arxiv.org/pdf/1609.08144v2.pdf)

[the](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Gap](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[between](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Human](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[and](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Machine](https://arxiv.org/pdf/1609.08144v2.pdf)[ ](https://arxiv.org/pdf/1609.08144v2.pdf)[Translation](https://arxiv.org/pdf/1609.08144v2.pdf)

Learning

This project allowed us to explore the domain of Question

Answering, an important task in Natural Language Process-

