# Natural-Language-Processing

This repository shows my NLP personal project and assignments. 
The files : albert.py, bidaf.py, qanet.py, roberta.py and copy_of_nlpprojectanalysis.py all are components of projects.

Rest of the files have smaller tasks covered which are explained later.

## Lets take a deeper look in project

### Intro:
The Stanford Question Answering Dataset (SQuAD 1.1),
is a reading comprehension dataset consisting of 100,000+
questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text
from the corresponding reading passage.
The Wikipedia articles were selected from project
Nayuki’s Wikipedia’s internal PageRanks to obtain the top
10000 articles of English Wikipedia, from which 536 articles were sampled uniformly at random. Paragraphs with
less than 500 characters, tables, and images were discarded.
The result was 23,215 paragraphs for the 536 articles covering a wide range of topics, from musical celebrities to abstract concepts.
On each paragraph, crowdworkers were tasked with asking and answering up to 5 questions on the content of that
paragraph Among the answers the shortest span in the paragraph that answered the question was selected. On average
4.8 answers were collected per question.
A major issue with SQuAD 1.1 was that the models only
needed to select the span that seemed most related to the
question, instead of checking that the answer is actually entailed by the text. To fix the issue the SQuAD 2.0 dataset
was released which combines existIng SQuAD data with
over 50,000 unanswerable questions written adversarially by
crowdworkers to look similar to answerable ones. This ensures that models must not only answer questions when possible but also determine when no answer is supported by the
paragraph and abstain from answering.

###Analysis
In the SQuAD2.0 dataset, we have tried to explore 75 percent of the training data. We wanted to come up with the
answer to “type” of questions, answers, and context alongside numbers. Upon further analysis, we came up with concepts of LDA (Latent Dirichlet Allocation) but LDA required training which would mean we need to have labeled
data. We have unlabeled data and now this could be a project
in itself to cluster unlabeled datasets. We also came across
different methods to deal with the problem of unlabeled data,
some of them were BERT-based models that could deal with
semi unlabeled data but going that way would mean deviation from our current project title. Just for data exploration, we required something pretrained and generalized
enough in a way that could be just like having an optimal
N number of clusters of unlabeled data. Spacy name entity
recognition seemed a perfect choice for this task as it’s pretrained and seems sophisticated enough for the task of data
exploration. We used Spacy’s “en core web sm” library as
its smaller than other two available and yet has similar F1
rates. It has 18 types of data categories namely ‘PERSON’,
’NORP’, ’FAC’, ’ORG’, ’GPE’, ’LOC’, ’PRODUCT’,
’EVENT’, ’WORK OF ART’, ’LAW’, ’LANGUAGE’,
’DATE’, ’TIME’, ’PERCENT’, ’MONEY’, ’QUANTITY’,
’ORDINAL’, ’CARDINAL’. For the task of type of question asked, we passed context, title, and combined question answer to Spacy NER algorithm. We found that Spacy
doesn’t work with single entities i.e. if we pass just a person’s name it returns None, as it is based upon the grammar
of the language and just one word could mean a noun, adjective, or anything. We then generated title-labels from the
context labels if the context contained the title within. That’s
how we came up with title labels. Then we passed combined
question and answers hoping that this way it can return much
better labels for question-answer and we were successful.
Next, we tried to find the number of each label type in titles
and for each of those titles, the number of questions type.
We found from the following plot that the majority of titles
were of three types i.e. Person, organization, GPE (countries, states, cities)
