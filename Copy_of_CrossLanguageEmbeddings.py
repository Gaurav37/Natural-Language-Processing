#!/usr/bin/env python
# coding: utf-8

# # Cross-Language Word Embeddings
# 
# We have mentioned, and will discuss in more detail this week, how we can reduce the dimensionality of word representations from their original vectors space to an embedding space on the order of a few hundred dimensions. Different modeling choices for word embeddings may be ultimately evaluated by the effectiveness of classifiers, parsers, and other inference models that use those embeddings.
# 
# In this assignment, however, we will consider another common method of evaluating word embeddings: by judging the usefulness of pairwise distances between words in the embedding space.
# 
# Follow along with the examples in this notebook, and implement the sections of code flagged with **TODO**.

# In[1]:


import gensim
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# We'll start by downloading a plain-text version of the Shakespeare plays we used for the first assignment.

# In[2]:


get_ipython().system('wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/shakespeare_plays.txt')
lines = [s.split() for s in open('shakespeare_plays.txt')]


# Then, we'll estimate a simple word2vec model on the Shakespeare texts.

# In[3]:


model = Word2Vec(lines)


# Even with such a small training set size, you can perform some standard analogy tasks.

# In[4]:


model.wv.most_similar(positive=['king','woman'], negative=['man'])


# For the rest of this assignment, we will focus on finding words with similar embeddings, both within and across languages. For example, what words are similar to the name of the title character of *Othello*?

# In[5]:


model.wv.most_similar(positive=['othello'])
#model.wv.most_similar(positive=['brutus'])


# This search uses cosine similarity. In the default API, you should see the same similarity between the words `othello` and `desdemona` as in the search results above.

# In[6]:


model.wv.similarity('othello', 'desdemona')


# **TODO**: Your **first task**, therefore, is to implement your own cosine similarity function so that you can reuse it outside of the context of the gensim model object.

# In[8]:


## TODO: Implement cosim
def cosim(v1, v2):
  ## return cosine similarity between v1 and v2
	dot_product = np.dot(v1, v2)
	norm_v1 = np.linalg.norm(v1)
	norm_v2 = np.linalg.norm(v2)
	return dot_product / (norm_v1 * norm_v2)  
  

## This should give a result similar to model.wv.similarity:
cosim(model.wv['othello'], model.wv['desdemona'])# model.wv creates vector of two 


# ## Evaluation
# 
# We could collect a lot of human judgments about how similar pairs of words, or pairs of Shakespearean characters, are. Then we could compare different word-embedding models by their ability to replicate these human judgments.
# 
# If we extend our ambition to multiple languages, however, we can use a word translation task to evaluate word embeddings.
# 
# We will use a subset of [Facebook AI's FastText cross-language embeddings](https://fasttext.cc/docs/en/aligned-vectors.html) for several languages. Your task will be to compare English both to French, and to *one more language* from the following set: Arabic, German, Portuguese, Russian, Spanish, Vietnamese, and Chinese.

# In[9]:


get_ipython().system('wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.en.vec')
get_ipython().system('wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.fr.vec')

# TODO: uncomment at least one of these to work with another language
get_ipython().system('wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.ar.vec')
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.de.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.pt.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.ru.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.es.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.vi.vec
# !wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/30k.zh.vec


# We'll start by loading the word vectors from their textual file format to a dictionary mapping words to numpy arrays.

# In[10]:


def vecref(s):
  (word, srec) = s.split(' ', 1)
  return (word, np.fromstring(srec, sep=' '))

def ftvectors(fname):
  return { k:v for (k, v) in [vecref(s) for s in open(fname)] if len(v) > 1} 

envec = ftvectors('30k.en.vec')
frvec = ftvectors('30k.fr.vec')

# TODO: load vectors for one more language, such as zhvec (Chinese)
arvec = ftvectors('30k.ar.vec')
#devec = ftvectors('30k.de.vec')
# ptvec = ftvectors('30k.pt.vec')
# ruvec = ftvectors('30k.ru.vec')
# esvec = ftvectors('30k.es.vec')
# vivec = ftvectors('30k.vi.vec')
# zhvec = ftvectors('30k.zh.vec')


# **TODO**: Your next task is to write a simple function that takes a vector and a dictionary of vectors and finds the most similar item in the dictionary. For this assignment, a linear scan through the dictionary using your `cosim` function from above is acceptible.

# In[17]:


## TODO: implement this search function
def mostSimilar(vec, vecDict):
  ## Use your cosim function from above
  mostSimilar = ''
  similarity = 0
  for word in vecDict:
    simi=cosim(vec, vecDict[word])
    if simi>similarity:
      mostSimilar=word
      similarity=simi
  return (mostSimilar, similarity)

## some example searches
[mostSimilar(envec[e], frvec) for e in ['computer', 'germany', 'matrix', 'physics', 'yeast']]


# Some matches make more sense than others. Note that `computer` most closely matches `informatique`, the French term for *computer science*. If you looked further down the list, you would see `ordinateur`, the term for *computer*. This is one weakness of a focus only on embeddings for word *types* independent of context.
# 
# To evalute cross-language embeddings more broadly, we'll look at a dataset of links between Wikipedia articles.

# In[18]:


get_ipython().system('wget http://www.ccs.neu.edu/home/dasmith/courses/cs6120/links.tab')
links = [s.split() for s in open('links.tab')]


# This `links` variable consists of triples of `(English term, language, term in that language)`. For example, here is the link between English `academy` and French `académie`:

# In[36]:


links[302]
# for i in range(0,10):
#   print(links[i])
#a_list = ["a4cd", "12345", "argument", "a", "2"]
def get_lang_links(fr):
  fr_links_index = [idx for idx, element in enumerate(links) if element[1]==fr ]
  fr_links=[]
  for i in fr_links_index:
    fr_links.append(links[i])
  return fr_links


# **TODO**: Evaluate the English and French embeddings by computing the proportion of English Wikipedia articles whose corresponding French article is also the closest word in embedding space. Skip English articles not covered by the word embedding dictionary. Since many articles, e.g., about named entities have the same title in English and French, compute the baseline accuracy achieved by simply echoing the English title as if it were French. Remember to iterate only over English Wikipedia articles, not the entire embedding dictionary.

# In[ ]:


## TODO: Compute English-French Wikipedia retrieval accuracy.
def Accuracy(fr_links,langvec):

  accuracy = 0
  tot_count=0
  count_sim=0
  for i in fr_links:
    if i[0] in envec.keys():
      fr_word, similarity=mostSimilar(envec[i[0]], langvec)
      tot_count+=1
      if fr_word==i[2]:
        count_sim+=1
  accuracy=count_sim/tot_count
  return accuracy
fr_links=get_lang_links('fr')
print(fr_links[0])
print(Accuracy(fr_links,frvec))


# In[54]:


fr_links=get_lang_links('fr')
print(fr_links[0])
def baselineAcc(fr_links):# Requires to calculate lang_links or fr_links from get_lang_links(fr)
  baselineAccuracy=0
  for i in fr_links:
    #print(i)
    if i[0]==i[2]:
      baselineAccuracy+=1
  baselineAccuracy=baselineAccuracy/len(fr_links)
  return baselineAccuracy
print(baselineAcc(fr_links))


# **TODO**: Compute accuracy and baseline (identity function) acccuracy for Englsih and another language besides French. Although the baseline will be lower for languages not written in the Roman alphabet (i.e., Arabic or Chinese), there are still many articles in those languages with headwords written in Roman characters.

# In[ ]:


## TODO: Compute English-X Wikipedia retrieval accuracy.
ar_links=get_lang_links('ar')
print("Baseline accuracy for Arabic",baselineAcc(ar_links))
print("Accuracy for Arabic is: ",Accuracy(ar_links,arvec) )


# Further evaluation, if you are interested, could involve looking at the $k$ nearest neighbors of each English term to compute "recall at 10" or "mean reciprocal rank at 10".
