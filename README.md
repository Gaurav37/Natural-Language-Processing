# Natural-Language-Processing

This repository shows my NLP personal project and assignments. 
The files : albert.py, bidaf.py, qanet.py, roberta.py and copy_of_nlpprojectanalysis.py all are components of projects.

Rest of the files have smaller tasks covered which are explained later.

## Lets take a deeper look in project

### Intro:
The Stanford Question Answering Dataset (SQuAD 1.1), is a reading comprehension dataset consisting of 100,000+ questions posed by crowd workers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.
The Wikipedia articles were selected from project Nayuki’s Wikipedia’s internal PageRanks to obtain the top 10000 articles of English Wikipedia, from which 536 articles were sampled uniformly at random. Paragraphs with less than 500 characters, tables, and images were discarded.
The result was 23,215 paragraphs for the 536 articles covering a wide range of topics, from musical celebrities to abstract concepts. On each paragraph, crowd workers were tasked with asking and answering up to 5 questions on the content of that paragraph Among the answers the shortest span in the paragraph that answered the question was selected. On average 4.8 answers were collected per question.
A major issue with SQuAD 1.1 was that the models only needed to select the span that seemed most related to the question, instead of checking that the answer is actually entailed by the text. To fix the issue the SQuAD 2.0 dataset was released which combines existIng SQuAD data with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. This ensures that models must not only answer questions when possible but also determine when no answer is supported by the paragraph and abstain from answering.

### Analysis
In the SQuAD2.0 dataset, we have tried to explore 75 percent of the training data. We wanted to come up with the answer to “type” of questions, answers, and context alongside numbers. Upon further analysis, we came up with concepts of LDA (Latent Dirichlet Allocation) but LDA required training which would mean we need to have labeled data. We have unlabeled data and now this could be a project in itself to cluster unlabeled datasets. We also came across different methods to deal with the problem of unlabeled data, some of them were BERT-based models that could deal with semi unlabeled data but going that way would mean deviation from our current project title. Just for data exploration, we required something pretrained and generalized enough in a way that could be just like having an optimal
N number of clusters of unlabeled data. Spacy name entity recognition seemed a perfect choice for this task as it’s pretrained and seems sophisticated enough for the task of data exploration. We used Spacy’s “en core web sm” library as its smaller than other two available and yet has similar F1 rates. It has 18 types of data categories namely ‘PERSON’, ’NORP’, ’FAC’, ’ORG’, ’GPE’, ’LOC’, ’PRODUCT’, ’EVENT’, ’WORK OF ART’, ’LAW’, ’LANGUAGE’, ’DATE’, ’TIME’, ’PERCENT’, ’MONEY’, ’QUANTITY’, ’ORDINAL’, ’CARDINAL’. For the task of type of question asked, we passed context, title, and combined question answer to Spacy NER algorithm. 
We found that Spacy doesn’t work with single entities i.e. if we pass just a person’s name it returns None, as it is based upon the grammar of the language and just one word could mean a noun, adjective, or anything. We then generated title-labels from the context labels if the context contained the title within. That’s how we came up with title labels. Then we passed combined question and answers hoping that this way it can return much better labels for question-answer and we were successful.
Next, we tried to find the number of each label type in titles and for each of those titles, the number of questions type. We found from the following plot that the majority of titles were of three types i.e. Person, organization, GPE (countries, states, cities)

![NLP project1](https://user-images.githubusercontent.com/19994641/200920674-29f8ede3-6beb-493c-9ad5-758fb90e2f24.png)

After this, we tried to plot the question types for all these categories of title types and came up with the following plot. It shows that most of the questions for each label type ask about persons, organizations, places, dates, and numbers which do not fall under other categories. Some relationships seen were: Norp (Nationality, religious groups) links with GPE (places, countries, cities, etc.) and dates. Organizations link with Person, organization, places, dates, ranks, and other kinds of numbers.

![NLP project2](https://user-images.githubusercontent.com/19994641/200920945-65ccf654-bb99-4ac0-84ae-af26770de34b.png)

After this, we tried to analyze our data from the length of context, questions and answers perspective and we found the following figures suggesting that most answers are of length 6, while most contexts are of length 500-600 characters and most questions have a length of 110 approximately. One point to be noted is that we tried to normalize this data by choosing distinct values of all contexts, questions, and answers.

![NLP project3](https://user-images.githubusercontent.com/19994641/200921754-2e962745-d797-454e-8229-5ef0d4bfaa5c.png)
![NLP project4](https://user-images.githubusercontent.com/19994641/200921827-63865472-798a-45da-83c2-1c740e39ac2d.png)
![NLP project5](https://user-images.githubusercontent.com/19994641/200921889-1c62f97e-7362-4bf0-a4da-8bacb2091bd3.png)

### Models
#### Bidaf
The paper [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1804.09541) implements is a multi-stage hierarchical architecture that represents the context and query at multiple levels of granularity. It also involves recurrence as it extensively uses LSTMs and a memory-less attention mechanism which is bi-directional in nature. Below are some important points regarding the BiDAF model. The key issue that this paper tries to address is that of early summarization in all the earlier approches that use attention mechanisms. The attention mechanisms until then were used to obtain a fixed-size summarization of given values and query. This, according to the authors leads to early summarization and loss of information. Moreover, previously, attention was only calculated in only one direction. To improve upon these issues, the authors propose a hierarchical, multi-stage network. Word embedding layers maps each word to a high dimensional vector space. We use pre-trained word vectors, GloVe to obtain the fixed word embedding of each word. A character embedding is calculated for each context and query word. This is done by using convolutions. It maps each word to a vector space using character-level CNNs. Here, we trained a simple CNN with one layer of convolution on top of pretrained word vectors and hypothesized that these pretrained word vectors could work as a universal feature extractors for various classification tasks. This is analogous to the earlier layers of vision models like VGG and Inception working as generic feature extractors. The intuition is simple over here. Just as convolutional filters learn various features in an image by operating on its pixels, here they’ll do so by operating on characters of words. Highway networks were originally introduced to ease the training of deep neural networks. The purpose of this layer is to learn to pass relevant information from the input. A high way network is a series of feed-forward or linear layers with a gating mechanism. The gating is implemented by using a sigmoid function which decides what amount of information should be transformed and what should be passed as it is. The input to this layer is the concatenation of word and character embeddings of each word. The idea here is that the adding of highway layers enables the network to make more efficient use of character embeddings. If a particular word is not found in the pretrained word vector vocabulary (OOV word), it will most likely be initialized with a zero vector. It then makes much more sense to look at the character embedding of that word rather than the word embedding. The soft gating mechanism in highway layers helps the model to achieve this. Attention Flow Layer is responsible for fusing and linking the context and query representations. This layer calculates attention in two directions: from context to query and from query to context. Attention vectors for these calculations are derived from a common matrix which is called as the similarity matrix. Modeling Layer is responsible for capturing temporal features interactions among the context words. This is done using a bidirectional LSTM. The difference between this layer and the contextual layer, both of which involve an LSTM layer is that here we have a query aware representation of the context while in the contextual layer, encoding of the context and query was independent. Contextual Embedding is the final embedding layer in the model. The output of highway layers is passed to a bidirection LSTM to model the temporal features of the text. This is done for both, the context and the query. The diagram of the BiDAF model is shown below.The paper [1804.09541] QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension (arxiv.org) implements is a multi-stage hierarchical architecture that represents the context and query at multiple levels of granularity. It also involves recurrence as it extensively uses LSTMs and a memory-less attention mechanism which is bi-directional in nature. Below are some important points regarding the BiDAF model. The key issue that this paper tries to address is that of early summarization in all the earlier approches that use attention mechanisms. The attention mechanisms until then were used to obtain a fixed-size summarization of given values and query. This, according to the authors leads to early summarization and loss of information. Moreover, previously, attention was only calculated in only one direction. To improve upon these issues, the authors propose a hierarchical, multi-stage network. Word embedding layers maps each word to a high dimensional vector space. We use pre-trained word vectors, GloVe to obtain the fixed word embedding of each word. A character embedding is calculated for each context and query word. This is done by using convolutions. It maps each word to a vector space using character-level CNNs. Here, we trained a simple CNN with one layer of convolution on top of pretrained word vectors and hypothesized that these pretrained word vectors could work as a universal feature extractors for various classification tasks. This is analogous to the earlier layers of vision models like VGG and Inception working as generic feature extractors. The intuition is simple over here. Just as convolutional filters learn various features in an image by operating on its pixels, here they’ll do so by operating on characters of words. Highway networks were originally introduced to ease the training of deep neural networks. The purpose of this layer is to learn to pass relevant information from the input. A high way network is a series of feed-forward or linear layers with a gating mechanism. The gating is implemented by using a sigmoid function which decides what amount of information should be transformed and what should be passed as it is. The input to this layer is the concatenation of word and character embeddings of each word. The idea here is that the adding of highway layers enables the network to make more efficient use of character embeddings. If a particular word is not found in the pretrained word vector vocabulary (OOV word), it will most likely be initialized with a zero vector. It then makes much more sense to look at the character embedding of that word rather than the word embedding. The soft gating mechanism in highway layers helps the model to achieve this. Attention Flow Layer is responsible for fusing and linking the context and query representations. This layer calculates attention in two directions: from context to query and from query to context. Attention vectors for these calculations are derived from a common matrix which is called as the similarity matrix. Modeling Layer is responsible for capturing temporal features interactions among the context words. This is done using a bidirectional LSTM. The difference between this layer and the contextual layer, both of which involve an LSTM layer is that here we have a query aware representation of the context while in the contextual layer, encoding of the context and query was independent. Contextual Embedding is the final embedding layer in the model. The output of highway layers is passed to a bidirection LSTM to model the temporal features of the text. This is done for both, the context and the query. The diagram of the BiDAF model is shown below.

<img width="875" alt="bidaf" src="https://user-images.githubusercontent.com/19994641/200925221-e0741b65-d59e-4b68-9219-3f1060eedaa9.png">

