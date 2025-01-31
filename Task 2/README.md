### Markdown version of the jupyter notebook

```python
from typing import List, Union, Dict, Set, Tuple, Sequence
from numpy.typing import NDArray
import sys
```

### Task 2.1: WordNet word similarity

In this task, we want to implement the similarity between two words in WordNet (https://www.nltk.org/api/nltk.corpus.reader.wordnet.html) using the NLTK package. The word similarity between two words is given by
$$
\frac{1}{1+d}
$$
where $d$ is the distance of the shortest path in the hypernym/hyponym hierarchy tree in WordNet between any pair of synsets that are associated with the two input words.

From NLTK you should __only__ use the `synsets`, `hypernyms` and `instance_hpyernyms` functions.

The following subtasks build on each other, i.e. the functions of the preceding subtasks can be used for the current subtask.

_Note: the distance of a synset to itself is 0, the distance to a direct hypernym is 1, ..._


```python
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
```

__a)__ Write a function ``shortest_paths_to`` that takes a synset as input and computes the shortest paths to all nodes on the way to the root in the hypernym hierarchy tree of WordNet. The function should return a dictionary that matches all visited hypernyms on the way(s) to the root to the distance of the shortest path from the input synset. Consider that a synset might have multiple paths to the root and that some nodes might appear in multiple paths. However, we only want to store the shortest distances. Moreover, keep in mind that the input synset might be an instance. 

Use the signature in the cell below.

__Example:__ _Calling_ ``shortest_paths_to(s)`` _on the synset_ ``s = wn.synset('calculator.n.01')`` _should yield the following result:_

``
{Synset('calculator.n.01'): 0,
 Synset('expert.n.01'): 1,
 Synset('person.n.01'): 2,
 Synset('causal_agent.n.01'): 3,
 Synset('organism.n.01'): 3,
 Synset('physical_entity.n.01'): 4,
 Synset('living_thing.n.01'): 4,
 Synset('entity.n.01'): 5,
 Synset('whole.n.02'): 5,
 Synset('object.n.01'): 6}
``


```python
def path_travel(syn,count,word_dict):
    
    word_dict[str(syn)] = count if word_dict.get(str(syn)) == None else min(word_dict[str(syn)],count)   # isn't the min function useless? because if an existing count exists, it will alwazs be lesser than the curent one. As count is only incremented, never decremented 
    nodes = syn.hypernyms()
    if len(nodes) == 0:
        return word_dict
    else:
        count+=1
        for i in nodes:
            word_dict = path_travel(i,count,word_dict)
        return word_dict
            


def shortest_paths_to(start_syn: Synset) -> Dict[Synset, int]:
    """Compute the shortest distance to all nodes on paths to the root.
    :param start_syn: synset to which we want to compute the shortest distances
    :return: dict that matches all visited hypernyms to their distance to the input synset
    """
    shortest_path = {}
    shortest_path = path_travel(start_syn,0,shortest_path) 
    
    return shortest_path

```

__b)__ Write a function ``merge_paths`` that gets two dictionaries that map synsets to shortest distances (you can assume they were created by the function from __a)__) and merges them. The function shold return a dictionary that includes all synsets and distances that appear in any of the input dictionaries. If a synset appears in both input dictionaries, we want to keep the shorter distance. Leave the input dictionaries unaltered.

Use the signature in the cell below.


```python
def merge_paths(p1: Dict[Synset, int], p2: Dict[Synset, int]) -> Dict[Synset, int]:
    """Merge two paths keeping the shorter distance for synsets that appear more than once.
    :param p1: first dict that maps synsets to their shortest distances
    :param p2: second dict that maps synsets to their shortest distances
    :return: merged dict
    """
    mergedDict = {}
    for key in p1:
        if p2.get(key):
            mergedDict[key]=min(p1.get(key), p2.get(key))
        else:
            mergedDict[key] = p1[key]

    for key in p2:
        if not (mergedDict.get(key)):
            mergedDict[key]=p2.get(key)
    
    return mergedDict

```

Test of the functions untill now


```python
test1 = shortest_paths_to(wn.synsets("car")[0])
print(test1)
```

    {"Synset('car.n.01')": 0, "Synset('motor_vehicle.n.01')": 1, "Synset('self-propelled_vehicle.n.01')": 2, "Synset('wheeled_vehicle.n.01')": 3, "Synset('container.n.01')": 4, "Synset('instrumentality.n.03')": 5, "Synset('artifact.n.01')": 6, "Synset('whole.n.02')": 7, "Synset('object.n.01')": 8, "Synset('physical_entity.n.01')": 9, "Synset('entity.n.01')": 10, "Synset('vehicle.n.01')": 4, "Synset('conveyance.n.03')": 5}



```python
test2=shortest_paths_to(wn.synsets("car")[1])
print(test2)
```

    {"Synset('car.n.02')": 0, "Synset('wheeled_vehicle.n.01')": 1, "Synset('container.n.01')": 2, "Synset('instrumentality.n.03')": 3, "Synset('artifact.n.01')": 4, "Synset('whole.n.02')": 5, "Synset('object.n.01')": 6, "Synset('physical_entity.n.01')": 7, "Synset('entity.n.01')": 8, "Synset('vehicle.n.01')": 2, "Synset('conveyance.n.03')": 3}



```python
merge_paths(test1, test2)
```




    {"Synset('car.n.01')": 0,
     "Synset('motor_vehicle.n.01')": 1,
     "Synset('self-propelled_vehicle.n.01')": 2,
     "Synset('wheeled_vehicle.n.01')": 1,
     "Synset('container.n.01')": 2,
     "Synset('instrumentality.n.03')": 3,
     "Synset('artifact.n.01')": 4,
     "Synset('whole.n.02')": 5,
     "Synset('object.n.01')": 6,
     "Synset('physical_entity.n.01')": 7,
     "Synset('entity.n.01')": 8,
     "Synset('vehicle.n.01')": 2,
     "Synset('conveyance.n.03')": 3,
     "Synset('car.n.02')": 0}



__c)__ Write a function ``all_hypernym_paths`` that gets a word as input and returns a dictionary that maps all hypernyms that are reachable from the set of synsets associated with the word to the shortest distance leading there.

Use the signature in the cell below.


```python
def all_hypernym_paths(word: str) -> Dict[Synset, int]:
    """Get all hypernyms of all synsets associated with the input word and compute the shortest distance leading there.
    :param word: input word
    :return: dict that matches all reachable hypernyms to their shortest distance
    """
    
    all_associated_synsets = wn.synsets(word)
    paths= {}
    for syn in all_associated_synsets:
        dict_paths = shortest_paths_to(syn)
        paths = merge_paths(paths, dict_paths)
    

    return paths 

```

__d)__  Write a function ``get_dist`` that returns the word similarity between two input words, according to the formula given in the task description at the beginning.  

Use the signature in the cell below.


```python
def get_dist(w1: str, w2: str) -> float:
    """Compute the similarity between two input words in the WordNet hierarchy tree.
    :param w1: first input word
    :param w2: second input word
    :return: word similarity
    """
    all_hypernym_paths_w1 = all_hypernym_paths(w1)
    all_hypernym_paths_w2 = all_hypernym_paths(w2)
    
    keys_intersection = set(all_hypernym_paths_w1.keys()).intersection(all_hypernym_paths_w2.keys()) #common hypernyms 
    
    if not keys_intersection:
        return 0.0
    
    min_distance = sys.maxsize
    for hypernym in keys_intersection:
        distance = all_hypernym_paths_w1[hypernym] + all_hypernym_paths_w2[hypernym]
        if distance < min_distance:
            min_distance = distance
        
    
    return 1 / (1 + min_distance)

get_dist("truck", "car")

```




    0.3333333333333333



### Task 2.2: Lesk algorithm

In this task we want to implement a simple version of the Lesk algorithm, a thesaurus-based method for word sense disambiguation. Given a target word $w$ and a context, the algorithm finds the word sense that is most fitting in the context. To achieve this, the Lesk algorithm computes the number of overlapping words between the context sentence and the definitions of the WordNet synsets, associated with $w$.

Write a function ``lesk`` that takes a word and a context string (e.g. a sentence) and returns the most fitting sense from the synsets associated with the word and the corresponding context overlap. The most fitting sense is the one whose definition shares the most words with the context string. Before matching tokens, make sure to 

* only include valid tokens (cf. HA 1, task 2a)
* remove stopwords
* only match stems of words (e.g. consider the ``PorterStemmer`` from ``nltk``)

When computing the context overlap, count each stemmed word only once, even if they occur multiple times. If there is no fitting synset, i.e. the context overlap between the context and the synset definitions is 0, return None instead.

Use the signature in the cell below.


```python
# HA 1, task 2a)
from nltk.corpus.reader.util import StreamBackedCorpusView
from nltk.corpus import stopwords
import re
import string

def get_valid_tokens(tokens: Union[List[str], StreamBackedCorpusView], remove_stopwords: bool=False) -> List[str]:
    """
    :param tokens: list of tokens that should be cleaned
    :param remove_stopwords: bool indicating if stopwords should be removed
                             False by default
    :return: list of valid tokens
    """
    valid = []
    punct = string.punctuation
    stop = stopwords.words('english')
    digit = re.compile(r"\d+")

    for t in tokens:
        if t in punct:
            continue
        if remove_stopwords and t.lower() in stop:
            continue
        if re.fullmatch(digit, t):
            continue
        valid.append(t.lower())
    return valid
```


```python
def lesk(word: str, context: str) -> (Synset, int):
    '''
    Compute the most probable sense of a word in the given context.
    :param word: ambiguous word
    :param context: context in which the word appears
    :returns:
        - synset with the most likely word sense
        - context overlap of synset and context
    '''
    # your code here

    tokens=get_valid_tokens(word_tokenize(context), True)
    porter=PorterStemmer()
    stems_context=set() #we need to count each word only once (uniqueness)
    for token in tokens:
      stems_context.add(porter.stem(token))

    synsets=wn.synsets(word) #get the synsets for the given word

    if not synsets: #synset not found
      return None,0

    elif synsets: #synset found
      overlap_max=0
      bestSS=None

      for synset in synsets:

        #Take definiton for the target word
        definition = synset.definition() 
        definition_tokens = nltk.word_tokenize(definition)
        definition_tokens = get_valid_tokens(definition_tokens,True)
        stems_definition = set(porter.stem(token) for token in definition_tokens) 

        # Take example for each synset to make the algorithm work better
        example = ' '.join(synset.examples())
        example_tokens = word_tokenize(example)
        example_tokens = get_valid_tokens(example_tokens,True)
        stems_example = set(porter.stem(token) for token in example_tokens)
        
        # Join stems from context and definition+examples
        stems_all = stems_definition.union(stems_example)
        
        overlap = len(stems_context & stems_all) #calculate overlap

        if overlap > overlap_max:
            overlap_max = overlap
            bestSS = synset
        
    if overlap_max==0:
      return None,0

    return bestSS, overlap_max
```


```python
lesk("bank", "i collect sea shells from the bank")[0].definition()
```




    'sloping land (especially the slope beside a body of water)'




```python
lesk("bank", "i collect money from the bank")[0].definition()
```




    'a financial institution that accepts deposits and channels the money into lending activities'



### Task 2.3: Minimum cost string alignment

In this tak we want to compute an alignment between two strings, that has minimum edit distance. 

Implement a function that takes two strings and their edit distance matrix and returns a minimum cost alignment. You can assume that the edit distance matrix is provided by the function that you implemented in exercise 3, task 2, with a substitution cost of 2. 

A minimum cost alignment consists of two strings that, printed below each other comprise the alignment, where insertions and deletions are represented by a ``*``. Use the function signature in the cell below.

__Example:__ _Given the input strings_ `"INTENTION"` _and_ `"EXECUTION"` _and the corresponding edit distance matrix:_

```
[[ 0  1  2  3  4  5  6  7  8  9]
 [ 1  2  3  4  5  6  7  6  7  8]
 [ 2  3  4  5  6  7  8  7  8  7]
 [ 3  4  5  6  7  8  7  8  9  8]
 [ 4  3  4  5  6  7  8  9 10  9]
 [ 5  4  5  6  7  8  9 10 11 10]
 [ 6  5  6  7  8  9  8  9 10 11]
 [ 7  6  7  8  9 10  9  8  9 10]
 [ 8  7  8  9 10 11 10  9  8  9]
 [ 9  8  9 10 11 12 11 10  9  8]]
```
  
_your function should return the two strings_ ``INTE***NTION`` _and_ ``***EXECUTION`` _that represent the alignment, when printed below each other:_
 
 ``INTE***NTION``    
 ``***EXECUTION`` 
 
 __Remark:__ _The alignment in the example above is not the only solution. In this task all alignments with minimum edit distance are accepted._



```python
import numpy as np

def compute_edit_distance_matrix(str1: str, str2: str) -> np.ndarray:
    m, n = len(str1), len(str2)
    D = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 2
            D[i][j] = min(D[i - 1][j - 1] + cost, D[i - 1][j] + 1, D[i][j - 1] + 1)
    
    return D

def get_alignment(str1: str, str2: str, D: np.ndarray) -> tuple[str, str]:
    i, j = len(str1), len(str2)
    aligned_str1, aligned_str2 = [], []
    
    while i > 0 or j > 0:
        # Handle the diagonal (substitution/match) case first
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if str1[i - 1] == str2[j - 1] else 2):
            aligned_str1.append(str1[i - 1])
            aligned_str2.append(str2[j - 1])
            i -= 1
            j -= 1
        # Handle the insertion case
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            aligned_str1.append('*')
            aligned_str2.append(str2[j - 1])
            j -= 1
        # Handle the deletion case
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            aligned_str1.append(str1[i - 1])
            aligned_str2.append('*')
            i -= 1
    
    return ''.join(reversed(aligned_str1)), ''.join(reversed(aligned_str2))

str1 = "INTENTION"
str2 = "EXECUTION"
D = compute_edit_distance_matrix(str1, str2)
aligned_str1, aligned_str2 = get_alignment(str1, str2, D)
print(aligned_str1)
print(aligned_str2)
```

    INTE*NTION
    *EXECUTION

