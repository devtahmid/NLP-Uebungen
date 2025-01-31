### Markdown version of the jupyter notebook



```python
from typing import List, Union, Dict, Set, Tuple
from numpy.typing import NDArray
from collections import defaultdict, Counter
import numpy as np
import re
from math import log
```

### Task 1: ML Basics - Naive Bayes Classification 
In this task, we want to build a Naive Bayes classifier with add-1 smoothing for text classification (pseudocode given below), e.g., to assign a category to a document. Use the class-skeleton provided below for your implementation.

#### Naive Bayes Pseudocode
##### TrainMultiNomialNB($\mathbb C$,$\mathbb D$)  
$V \leftarrow extractVocabulary(\mathbb D)$  
$N \leftarrow countDocs(\mathbb D)$    
for $c \in \mathbb C$:  
&nbsp;&nbsp;&nbsp;&nbsp;$N_c \leftarrow countDocsInClass(\mathbb D, c)$  
&nbsp;&nbsp;&nbsp;&nbsp;$prior[c] \leftarrow \frac{N_c}{N}$  
&nbsp;&nbsp;&nbsp;&nbsp;$text_c \leftarrow concatenateTextOfAllDocsInClass(\mathbb D, c)$   
&nbsp;&nbsp;&nbsp;&nbsp;for $t \in V$:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$T_{ct} \leftarrow countTokensOfTerm(text_c,t)$  
&nbsp;&nbsp;&nbsp;&nbsp;for $t \in V$:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$condprob[t][c] \leftarrow \frac{T_{ct} + 1}{\sum_{t'}(T_{ct'} + 1)}$  
return $V,prior,condprob$

##### ApplyMultinomialNB($\mathbb C,V,prior,condprob,d$)
$W \leftarrow extractTokensFromDoc(V,d)$   
for $c \in \mathbb C$:  
&nbsp;&nbsp;&nbsp;&nbsp;$score[c] \leftarrow log(prior[c])$  
&nbsp;&nbsp;&nbsp;&nbsp;for $t \in W$:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$score[c] += log(condprob[t][c])$  
return $argmax_{c \in \mathbb C} score[c]$

__a) Tokenization__  
Implement the function `tokenize` to transform a text document to a list of tokens with the regex pattern `\b\w\w+\b`. Transform all tokens to lowercase.

__b) Naive Bayes "Training"__  
Implement the `__init__` function to set up the Naive Bayes Model. Cf. TrainMultiNomialNB($\mathbb C$,$\mathbb D$) in the pseudocode above. Contrary to the pseudocode, the `__init__` function should not return anything, but the vocabulary, priors and conditionals should be stored in class variables. We only want to keep tokens with a frequeny >= `min_count` in the vocabulary.

__c) Naive Bayes Classification__  
Implement the `classify` function to return the most probable class for the provided document according to your Naive Bayes model.


```python
class NaiveBayesClassifier:
    '''Naive Bayes for text classification.'''
    def __init__(self, docs: List[str], labels: List[int], min_count: int=1):
        '''
        :param docs: list of documents from which to build the model (corpus)
        :param labels: list of classes assigned to the list of documents (labels[i] is the class for docs[i])
        :param min_count: minimum frequency of token in vocabulary (tokens that occur less times are discarded)
        '''
        # your code for Task 1b) here
        self.vocab = set()
        self.priors = {}
        self.cond_probs = {}
        self.class_word_counts = defaultdict(Counter)
        self.class_doc_counts = Counter(labels)
        self.total_docs = len(docs)
        
        tokenized_docs = [self.tokenize(doc) for doc in docs]
        
        for label in self.class_doc_counts:
            self.priors[label] = self.class_doc_counts[label] / self.total_docs
        
        for tokens, label in zip(tokenized_docs, labels):
            for token in tokens:
                self.class_word_counts[label][token] += 1

        total_word_counts = Counter()
        for class_counter in self.class_word_counts.values():
            total_word_counts.update(class_counter)
        self.vocab = {word for word, count in total_word_counts.items() if count >= min_count}

        self.cond_probs = defaultdict(dict)
        for label in self.class_word_counts:
            total_words = sum(self.class_word_counts[label].values())
            for word in self.vocab:
                word_count = self.class_word_counts[label][word]
                self.cond_probs[label][word] = (word_count + 1) / (total_words + len(self.vocab))

        # print("Conditional Propabilities : ",self.cond_probs)
        # print("class_word_counts", class_word_counts)

    def tokenize(self, doc: str):
        '''
        :param doc: document to tokenize
        :return: document as a list of tokens
        '''
        # your code for Task 1a) here
        tokens = re.findall(r'\b\w\w+\b', doc.lower())
        return tokens

    def classify(self, doc: str):
        '''
        :param doc: document to classify
        :return: most probable class
        '''
        # your code for Task 1c) here
        tokens = self.tokenize(doc)
        class_scores = {}

        for label in self.priors:
            log_prob = log(self.priors[label])
            for token in tokens:
                if token in self.vocab:
                    log_prob += log(self.cond_probs[label].get(token, 1 / (sum(self.class_word_counts[label].values()) + len(self.vocab))))
            class_scores[label] = log_prob

        return max(class_scores, key=class_scores.get)

```

__d) Evaluation__
Test your implementation on the 20newsgroups dataset. If implemented correctly, with `min_count=1` your Naive Bayes classifier should obtain the same accuracy as the implementation by scikit-learn (see below for comparison).


```python
# Using a subset of the data for evaluvation since the time taken for entire dataset is high.
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
min_count = 1
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
nb_classifier = NaiveBayesClassifier(train.data[:500], train.target[:500], 1)
pred = []
for i in test.data[:100]:
    # print(nb_classifier.classify(i))
    pred.append(nb_classifier.classify(i))
accuracy_score(test.target[:100],pred)
```




    0.23




```python
# Accuracy of the SKlearn Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# see https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html for details
from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
# print(train.target[:10])
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(train.data[:500])
clf = MultinomialNB()
clf.fit(x,train.target[:500])

pred = clf.predict(vectorizer.transform(test.data[:100]))

accuracy_score(test.target[:100],pred)
```




    0.23



### Task 2: Term Frequency - Inverse Document Frequency 

In this task we want to use the term frequency - inverse document frequency (tf-idf) weighting method to compare documents with each other and to queries. In the end, we will apply our method to a subset of wikipedia pages (more specifically: only the introduction sections) that are linked to from the English Wikipedia page of Mannheim.

In case you need to tokenize any sentences in the following tasks, please use a tokenizer from NLTK and not the ``string.split`` function.

__a)__ To test your implementation throughout this task, you are given the example from exercise 8. Start by implementing a function ``process_docs`` that takes the provided dictionary of documents and returns the following data structures. 

- ``word2index``: a dictionary that maps each word that appears in any document to a unique integer identifier starting at 0 
- ``doc2index``: a dictionary that maps each document name (here given as the dictionary keys) to a unique integer identifier starting at 0
- ``index2doc``: a dictionary that maps each document identifier to the corresponding document name (reverse to ``doc2index``)
- ``doc_word_vectors``: a dictionary that maps each document name to a list of word ids that indicate which words appeared in the document in their order of appearance. Words that appear multiple times must also be included multiple times.


```python
# example from exercise 8
d1 = "cold beer beach"
d2 = "ice cream beer beer"
d3 = "beach cold ice cream"
d4 = "cold beer frozen yogurt frozen beer"
d5 = "frozen ice ice beer ice cream"
d6 = "yogurt ice cream ice cream"

docs = {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5, "d6": d6}
```


```python
from nltk.tokenize import word_tokenize
def process_docs(docs: Dict[str, str]) -> tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    """
    :params docs: dict that maps each document name to the document content
    :returns:
        - word2index: dict that maps each word to a unique id
        - doc2index: dict that maps each document name to a unique id
        - index2doc: dict that maps ids to their associated document name
        - doc_word_vectors: dict that maps each document name to a list of word ids that appear in it
    """
    word2index = {}
    doc2index = {}
    index2doc = {}
    doc_word_vectors = {}

    for i,d in enumerate(docs.keys()):
        doc2index[d]=i
        index2doc[i]=d


    word_ind = 0
    for doc,text in docs.items():
        temp = word_tokenize(text.lower())
        words_in_doc = []
        for t in temp:
            if t not in word2index:
                word2index[t]=word_ind
                word_ind+=1
            words_in_doc.append(word2index[t])
        doc_word_vectors[doc]=words_in_doc
    
    return word2index,doc2index,index2doc,doc_word_vectors
```


```python
# The output for the provided example could look like this:

# word2index:
# {'cold': 0, 'beer': 1, 'beach': 2, 'ice': 3, 'cream': 4, 'frozen': 5, 'yogurt': 6}

# doc2index:
# {'d1': 0, 'd2': 1, 'd3': 2, 'd4': 3, 'd5': 4, 'd6': 5}

# index2doc
# {0: 'd1', 1: 'd2', 2: 'd3', 3: 'd4', 4: 'd5', 5: 'd6'}

# doc_word_vectors:
# {'d1': [0, 1, 2],
#  'd2': [3, 4, 1, 1],
#  'd3': [2, 0, 3, 4],
#  'd4': [0, 1, 5, 6, 5, 1],
#  'd5': [5, 3, 3, 1, 3, 4],
#  'd6': [6, 3, 4, 3, 4]}

#Priniting the output of process_doc to check if the desired output is got.
data = process_docs(docs)
for i in data:
    print(i)
```

    {'cold': 0, 'beer': 1, 'beach': 2, 'ice': 3, 'cream': 4, 'frozen': 5, 'yogurt': 6}
    {'d1': 0, 'd2': 1, 'd3': 2, 'd4': 3, 'd5': 4, 'd6': 5}
    {0: 'd1', 1: 'd2', 2: 'd3', 3: 'd4', 4: 'd5', 5: 'd6'}
    {'d1': [0, 1, 2], 'd2': [3, 4, 1, 1], 'd3': [2, 0, 3, 4], 'd4': [0, 1, 5, 6, 5, 1], 'd5': [5, 3, 3, 1, 3, 4], 'd6': [6, 3, 4, 3, 4]}


__b)__ Set up a term-document matrix where each column corresponds to a document and each row corresponds to a word that was observed in any of the documents. The row/column indices should correspond to the word/document ids that are set in the input dicts ``word2index`` and ``doc2index``. Count how often each word appears in each document and fill the term document matrix. 

_Example: The word "beer" with the word id_ ``1`` _appears two times in the document "d4" that has the document id_ ``3``. _Therefore the the entry at position_ ``[1, 3]`` _in the term-document matrix is_ ``2``.




```python
import numpy as np
def term_document_matrix(doc_word_v: Dict[str, List[int]], doc2index: Dict[str, int], word2index: Dict[str, int]) -> NDArray[NDArray[float]]:
    """
    :param doc_word_v: dict that maps each document to the list of word ids that appear in it
    :param doc2index: dict that maps each document name to a unique id
    :param word2index: dict that maps each word to a unique id
    :return: term-document matrix (each word is a row, each document is a column) that indicates the count of each word in each document
    """
    tdm = np.zeros((len(word2index), len(doc2index)))
    for doc_name, word_ids in doc_word_v.items():
        doc_id = doc2index[doc_name]
        for word_id in word_ids:
            tdm[word_id, doc_id] += 1
    return tdm

```


```python
# Testing the term_document_matrix function
word2index, doc2index, index2doc, doc_word_vectors = process_docs(docs)
tdm = term_document_matrix(doc_word_vectors, doc2index, word2index)

print("Term-Document Matrix:")
print(tdm)
```

    Term-Document Matrix:
    [[1. 0. 1. 1. 0. 0.]
     [1. 2. 0. 2. 1. 0.]
     [1. 0. 1. 0. 0. 0.]
     [0. 1. 1. 0. 3. 2.]
     [0. 1. 1. 0. 1. 2.]
     [0. 0. 0. 2. 1. 0.]
     [0. 0. 0. 1. 0. 1.]]


__c)__ Implement the function ``to_tf_idf_matrix`` that takes a term-document matrix and returns the corresponding term frequency (tf) matrix. If the parameter ``idf`` is set to ``True``, the tf-matrix should further be transformed to a tf-idf matrix (i.e. every entry corresponds to the tf-idf value of the associated word and document). Your implementation should leave the input term-document matrix unchanged. 

Recall the following formulas:

\begin{equation}
  tf_{t,d} =
    \begin{cases}
      1+log_{10}\text{count}(t,d) & \text{if count}(t, d) > 0\\
      0 & \text{otherwise}
    \end{cases}       
\end{equation}  

$$idf_t = log_{10}(\frac{N}{df_i})$$  

$$tf-idf_{t,d} = tf_{t,d} \cdot idf_t$$


```python
def to_tf_idf_matrix(td_matrix: NDArray[NDArray[float]], idf: bool=True) -> NDArray[NDArray[float]]:
    """
    :param td_matrix: term-document matrix
    :param idf: computes the tf-idf matrix if True, otherwise computes only the tf matrix
    :return: matrix with tf(-idf) values for each word-document pair
    """
    num_words, num_docs = td_matrix.shape
    
    tf_matrix = np.zeros(td_matrix.shape)
    
    for doc in range(num_docs):
        for word in range(num_words):
            t = td_matrix[word][doc]
            if t > 0:
                tf_matrix[word, doc] = 1 + np.log10(t)
                

    if not idf:
        return tf_matrix

    doc_freq = np.count_nonzero(td_matrix, axis=1)
    idf_matrix = np.log10((num_docs  / doc_freq)) 

    # print(tf_matrix)
    # print(idf_matrix[:, None] )

    tf_idf_matrix = tf_matrix * idf_matrix[:, None] 

    return tf_idf_matrix

#Testing the to_tf_idf_matrix Function
# word2index, doc2index, index2doc, doc_word_vectors = process_docs(docs)
# tdm = term_document_matrix(doc_word_vectors, doc2index, word2index)
# # tfm = to_tf_idf_matrix(tdm,False)
# # print(tdm)
# # print("Term-Freq Matrix:")
# # print(tfm)
# print("TFIDF Matrix:")
# print(to_tf_idf_matrix(tdm))
```

__d)__ We want to test our implementation on our running example. First, print the tf-idf for each word of the query ``ice beer`` with respect to each document. Second, find the two most similar documents from ``d1, d2, d3`` according to cosine similarity and print all similarity values.  


```python
#using the cosine_similarity in sklearn to check the similarity of the vectors
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(query_vector: NDArray[float], doc_vectors: NDArray[NDArray[float]]) -> List[float]:
    similarities = cosine_similarity(query_vector.reshape(1, -1), doc_vectors.T).flatten()
    return similarities

tf_idf_matrix = to_tf_idf_matrix(tdm)
print("TF-IDF values for each word of the query 'ice beer' with respect to each document:")
query = "ice beer"

query_tokens = word_tokenize(query.lower())
query_vector = np.zeros(len(word2index))

for word in query_tokens:
    if word in word2index:
        word_id = word2index[word]
        print(f"Word: '{word}' (ID: {word_id})")
        for doc_id, doc_name in index2doc.items():
            print(f"Document: {doc_name}, TF-IDF: {tf_idf_matrix[word_id, doc_id]:.4f}")
        print()

similarity_with_query = compute_cosine_similarity(query_vector, tf_idf_matrix)
# print("Cosine similarity between the query 'ice beer' and each document:")

for doc_id, sim in enumerate(similarity_with_query):
    print(f"Document {index2doc[doc_id]}: {sim:.4f}")

doc_ids_to_compare = [doc2index["d1"], doc2index["d2"], doc2index["d3"]]
doc_vectors_to_compare = tf_idf_matrix[:, doc_ids_to_compare].T

# print(tf_idf_matrix[:, doc_ids_to_compare])
# print(tf_idf_matrix[:, doc_ids_to_compare].T)

similarity_matrix = cosine_similarity(doc_vectors_to_compare)
print("\nCosine similarity between documents d1, d2, and d3:")
for i, doc_id1 in enumerate(doc_ids_to_compare):
    for j, doc_id2 in enumerate(doc_ids_to_compare):
        if i < j:
            print(f"Similarity between {index2doc[doc_id1]} and {index2doc[doc_id2]}: {similarity_matrix[i, j]:.4f}")

# Find the two most similar documents
max_sim = -1
most_similar_docs = (None, None)
for i in range(len(doc_ids_to_compare)):
    for j in range(i + 1, len(doc_ids_to_compare)):
        if similarity_matrix[i, j] > max_sim:
            max_sim = similarity_matrix[i, j]
            most_similar_docs = (index2doc[doc_ids_to_compare[i]], index2doc[doc_ids_to_compare[j]])

print(f"\nThe two most similar documents are: {most_similar_docs[0]} and {most_similar_docs[1]} with a similarity of {max_sim:.4f}")

```

    TF-IDF values for each word of the query 'ice beer' with respect to each document:
    Word: 'ice' (ID: 3)
    Document: d1, TF-IDF: 0.0000
    Document: d2, TF-IDF: 0.1761
    Document: d3, TF-IDF: 0.1761
    Document: d4, TF-IDF: 0.0000
    Document: d5, TF-IDF: 0.2601
    Document: d6, TF-IDF: 0.2291
    
    Word: 'beer' (ID: 1)
    Document: d1, TF-IDF: 0.1761
    Document: d2, TF-IDF: 0.2291
    Document: d3, TF-IDF: 0.0000
    Document: d4, TF-IDF: 0.2291
    Document: d5, TF-IDF: 0.1761
    Document: d6, TF-IDF: 0.0000
    
    Document d1: 0.0000
    Document d2: 0.0000
    Document d3: 0.0000
    Document d4: 0.0000
    Document d5: 0.0000
    Document d6: 0.0000
    
    Cosine similarity between documents d1, d2, and d3:
    Similarity between d1 and d2: 0.2017
    Similarity between d1 and d3: 0.8733
    Similarity between d2 and d3: 0.2972
    
    The two most similar documents are: d1 and d3 with a similarity of 0.8733


__e)__ In a second step we want to find the documents that are most similar to a provided query. Therefore, implement the function ``process_query`` that creates a vector represention of the query. 

Create a vector that has an entry for each vocabulary word (words that appeared in any document), where the entry at position ``i`` indicates how often the word with id ``i`` (as indicated by ``word2index``) appears in the query. 

If ``tf`` is set to ``True``, you should transform all entries to tf-values. Similar, if ``idf`` is set to ``True``, return a vector with tf-idf values (cf. task __c)__). The computation of the idf values is based on the corresponding input term-document matrix.

In case the query contains words that are not in any of the documents, print an appropriate error message and stop the computation.


```python
def process_query(query: List[str], word2index: Dict[str, int], td_matrix: NDArray[NDArray[float]], tf: bool=True, idf: bool=True) -> NDArray[float]:
    """
    :param query: list of query tokens
    :param word2index: dict that maps each word to a unique id
    :param td_matrix: term-document matrix
    :param tf: computes the tf vector of the query if True
    :param idf: computes the tf-idf vector of the query if True, ignored if tf=False
    :return: vector representation of the input query (using tf(-idf))
    """
    num_words = len(word2index)
    
    query_vector = np.zeros(num_words, dtype=float)
    
    for token in query:
        if token in word2index:
            word_id = word2index[token]
            query_vector[word_id] += 1
        else:
            print(f"Error: Word '{token}' not found in the vocabulary.")
            return None
    
    if tf:
        for i in range(num_words):
            if query_vector[i] > 0:
                query_vector[i] = 1 + np.log10(query_vector[i])
    
    if idf:
        doc_freq = np.count_nonzero(td_matrix, axis=1)
        idf_values = np.log10((td_matrix.shape[1] / doc_freq ))
        query_vector = query_vector * idf_values
        
    return query_vector
query = ["ice", "beer"]
query_vector = process_query(query, word2index, tdm)

if query_vector is not None:
    print("Query vector (TF-IDF):")
    print(query_vector)
    similarity_with_query = compute_cosine_similarity(query_vector, tf_idf_matrix)
    print("Cosine similarity between the query and each document:")
    for doc_id, sim in enumerate(similarity_with_query):
        print(f"Document {index2doc[doc_id]}: {sim:.4f}")
else:
    print("Query processing failed due to unknown words.")
```

    Query vector (TF-IDF):
    [0.         0.17609126 0.         0.17609126 0.         0.
     0.        ]
    Cosine similarity between the query and each document:
    Document d1: 0.2107
    Document d2: 0.8467
    Document d3: 0.2019
    Document d4: 0.1863
    Document d5: 0.5160
    Document d6: 0.2809


__f)__ Implement a function ``most_similar_docs`` that gets the vector representation of a query (in terms of counts, tf values or tf-idf values) and a term-document matrix that can either contain counts, tf-values or tf-idf values.  Compute the cosine similarity between the query and all documents and return the document names and the cosine similarity values of the top-``k`` documents that are most similar to the query. The value of ``k`` should be specified by the user. 


```python
def most_similar_docs(query_v: NDArray[float], td_matrix: NDArray[NDArray[float]], index2doc: Dict[int, str], k: int) -> tuple[List[str], List[float]]:
    """
    :param query_v: vector representation of the input query
    :param td_matrix: term-document matrix, possibly with tf-(idf) values
    :param index2doc: dict that maps each document id to its name
    :k: number of documents to return
    :returns:
        - list with names of the top-k most similar documents to the query, ordered by descending similarity
        - list with cosine similarities of the top-k most similar docs, ordered by descending similarity
    """
    # your code here
    similarity_with_query = compute_cosine_similarity(query_vector, td_matrix)

    top_k_indices = np.argsort(similarity_with_query)[::-1][:k]
    top_k_docs = [index2doc[i] for i in top_k_indices]
    top_k_similarities = [similarity_with_query[i] for i in top_k_indices]
    
    return top_k_docs, top_k_similarities

query = ["ice", "beer"]
word2index, doc2index, index2doc, doc_word_vectors = process_docs(docs)
tdm = term_document_matrix(doc_word_vectors, doc2index, word2index)
tf_idf_matrix = to_tf_idf_matrix(tdm)

query_vector = process_query(query, word2index, tdm)

if query_vector is not None:
    top_k_docs, top_k_similarities = most_similar_docs(query_vector, tf_idf_matrix, index2doc, k=2)
    
    print("Top-k most similar documents:")
    for doc, sim in zip(top_k_docs, top_k_similarities):
        print(f"Document: {doc}, Similarity: {sim:.4f}")
else:
    print("Query processing failed due to unknown words.")
```

    Top-k most similar documents:
    Document: d2, Similarity: 0.8467
    Document: d5, Similarity: 0.5160


## Task 3: Text Classification 
In this task, we want to build a logistic regression classifier to classify 20newsgroups posts. As feature representation, we want to use tf-idf vectors as just implemented.

### Logistic Regression
Implement a logistic regression classifier, similar to exercise 7. Again, you don't need to add a bias weight/feature.

__a)__ Implement the `predict_proba` function in the `LogisticRegression` class below. Your function should return the output of a logistic regression classifier according to the current assignments of weights $\mathbf{w}$, i.e., 
$$
expit(\mathbf{w}^T\mathbf{x})
$$
You can assume that model weights are stored in a variable `self.w`. 

__b)__ Implement the `predict` function in the `LogisticRegression` class below. The prediction should return class `1` if the classifier output is above 0.5, otherwise `0` 

__c)__ Implement the `fit` function to learn the model parameters `w` with stochastic gradient descent for one epoch, i.e., going over the training data once. Store the learned parameters in a variable `self.w`. Only initialize the parameters randomly in the first training iteration and continue with learned parameters in later iterations. Make sure, that you iterate over instances in each epoch randomly.  



```python
# addeded epochs to the function parameter since its asked in question but was not present in the function name. The same has been used in exercise 7 

from scipy.special import expit

class LogisticRegression():
    '''Logistic Regression Classifier.'''
    def __init__(self):
        self.w = None

    def fit(self, x: NDArray[NDArray[float]], y: NDArray[int], eta: float=0.1, epochs: int=10):
        '''
        :param x: 2D numpy array where each row is an instance
        :param y: 1D numpy array with target classes for instances in x
        :param eta: learning rate, default is 0.1
        :param epochs: fixed number of epochs as stopping criterion, default is 10
        '''
        # c)
        if self.w is None:
            self.w = np.random.randn(x.shape[1])
        
        for epoch in range(epochs):

            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            
            for i in indices:
                xi = x[i]
                yi = y[i]

                proba = self.predict_proba(xi)
                
                self.w += eta * (yi - proba) * xi

    def predict_proba(self, x):
        # a)
        return expit(np.dot(self.w, x))

    def predict(self, x):
        # b)
        return np.where(self.predict_proba(x) > 0.5,1,0)
```

__d)__ Apply your model to the two categories 'comp.windows.x' and 'rec.motorcycles' from the 20newsgroups data. To this end, first transform the training data to tf-idf representation with the functions `process_docs`, `term_document_matrix` and `to_tfidf_matrix`. Transform the test documents with `process_query`. Fit your model on the training data for 10 epochs. Calculate the accuracy on the test data. 

**Shortcut**: use the `TfidfVectorizer` from scikit learn (you may need to transform its output to a dense (array) representation).


```python
# used the shortcutTfidfVectorizer

from sklearn.datasets import fetch_20newsgroups
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
train = fetch_20newsgroups(subset='train', categories=['comp.windows.x','rec.motorcycles'])
test = fetch_20newsgroups(subset='test', categories=['comp.windows.x','rec.motorcycles'])

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(train.data).toarray()
x_test_tfidf = vectorizer.transform(test.data).toarray()

y_train = np.array([1 if label == train.target_names.index('rec.motorcycles') else 0 for label in train.target])
y_test = np.array([1 if label == test.target_names.index('rec.motorcycles') else 0 for label in test.target])

model = LogisticRegression()
model.fit(x_train_tfidf, y_train, eta=0.1, epochs=10)

predictions = np.array([model.predict(x) for x in x_test_tfidf])

accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')

```

    Accuracy: 0.9066834804539723



```python

```


```python

```
