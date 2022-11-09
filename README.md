# Natural-Language-Processing

Question Answering or reading comprehension is the
ability to read a given text and then answer questions
about it. It is an important task based on which intelligence of NLP systems and AI in general can be
judged. This, is a challenging task for machines, requiring both understandings of natural language and knowledge about the world. The system must select the answer from all possible spans in the passage, thus needing to cope with a fairly large number of candidates.
In this project we attempt at implementing some of
the most important papers for Question Answering and
evaluate their performance. We compare different techniques for QA including training models from end to
end as well as fine tuning larger language models. To
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
from a human point of view, it remains to be a complex challenge for machines.
With the rise of deep learning-based models and more
and more public available datasets, we have seen significant progress on this task. In this project, we look at four
different approaches to solving the QA challenges. As the
attention mechanism has been indispensable in recent successful models. In this project, we investigate two such
attention-based models, BiDAF and QANet. We also look
at the more recent larger language models and investigate
two such flavors of pre-trained language models ALBERT
and RoBERTa.
Copyright © 2020, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
The structure of this report is as follows. We first begin
by introducing the dataset and do some exploratory analysis on the dataset. We then introduce the models and show
our experiments. Finally, we look at the learnings and future
work.
Dataset
The Stanford Question Answering Dataset (SQuAD 1.1),
is a reading comprehension dataset consisting of 100,000+
questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text
from the corresponding reading passage.
The Wikipedia articles were selected from project
Nayuki’s Wikipedia’s internal PageRanks to obtain the top
10000 articles of English Wikipedia, from which 536 articles were sampled uniformly at random. Paragraphs with
less than 500 characters, tables, and images were discarded.
The result was 23,215 paragraphs for the 536 articles covering a wide range of topics, from musical celebrities to abstract concepts.
On each paragraph, crowdworkers were tasked with asking and answering up to 5 questions on the content of that
paragraph Among the answers the shortest span in the paragraph that answered the question was selected. On average
4.8 answers were collected per question.
A major issue with SQuAD 1.1 was that the models only
needed to select the span that seemed most related to the
question, instead of checking that the answer is actually entailed by the text. To fix the issue the SQuAD 2.0 dataset
was released which combines existIng SQuAD data with
over 50,000 unanswerable questions written adversarially by
crowdworkers to look similar to answerable ones. This ensures that models must not only answer questions when possible but also determine when no answer is supported by the
paragraph and abstain from answering.
The dataset contains 130,319 questions in the Training set
and 11,873 questions in the Validation set along with 8,862
questions in the Test set. In our project, we will be using the
train and validation splits of the SQuAD 2.0 dataset (1). The
SQuAD dataset mentions the following metrics for evaluation and we will be evaluating performance on both of them.
Exact match
This metric measures the percentage of predictions that
match any one of the ground truth answers exactly.
F1 score (Macro-averaged)
This metric measures the average overlap between the prediction and ground truth answer. The prediction and ground
truth tokens are treated as bags of tokens and compute their
F1. The maximum F1 over all of the ground truth answers
for a given question is taken, and then averaged over all of
the questions.
Analysis
In the SQuAD2.0 dataset, we have tried to explore 75 percent of the training data. We wanted to come up with the
answer to “type” of questions, answers, and context alongside numbers. Upon further analysis, we came up with concepts of LDA (Latent Dirichlet Allocation) but LDA required training which would mean we need to have labeled
data. We have unlabeled data and now this could be a project
in itself to cluster unlabeled datasets. We also came across
different methods to deal with the problem of unlabeled data,
some of them were BERT-based models that could deal with
semi unlabeled data but going that way would mean deviation from our current project title. Just for data exploration, we required something pretrained and generalized
enough in a way that could be just like having an optimal
N number of clusters of unlabeled data. Spacy name entity
recognition seemed a perfect choice for this task as it’s pretrained and seems sophisticated enough for the task of data
exploration. We used Spacy’s “en core web sm” library as
its smaller than other two available and yet has similar F1
rates. It has 18 types of data categories namely ‘PERSON’,
’NORP’, ’FAC’, ’ORG’, ’GPE’, ’LOC’, ’PRODUCT’,
’EVENT’, ’WORK OF ART’, ’LAW’, ’LANGUAGE’,
’DATE’, ’TIME’, ’PERCENT’, ’MONEY’, ’QUANTITY’,
’ORDINAL’, ’CARDINAL’. For the task of type of question asked, we passed context, title, and combined questionanswer to Spacy NER algorithm. We found that Spacy
doesn’t work with single entities i.e. if we pass just a person’s name it returns None, as it is based upon the grammar
of the language and just one word could mean a noun, adjective, or anything. We then generated title-labels from the
context labels if the context contained the title within. That’s
how we came up with title labels. Then we passed combined
question and answers hoping that this way it can return much
better labels for question-answer and we were successful.
Next, we tried to find the number of each label type in titles
and for each of those titles, the number of questions type.
We found from the following plot that the majority of titles
were of three types i.e. Person, organization, GPE (countries, states, cities).
