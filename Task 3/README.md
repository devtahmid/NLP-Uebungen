### Markdown version of the jupyter notebook

```python
from typing import List, Union, Dict, Set, Tuple, Sequence
from numpy.typing import NDArray
import nltk
```

### Task 1: Markov chains

In this task we want to create a language model by using the independence assumption af Markov. We therefore assume that the probability of a word is only dependent on a fixed number of preceding words. E.g. by restricting the number of preceding words to $n$ we can approximate the probability of a word $w_{i}$ following a sequence of words $w_1, ..., w_{i-1}$ by:

$P(w_{i}|w_1, ..., w_{i-1}) \approx P(w_{i}|w_{i-n}, ..., w_{i-1})$

We will first train our model on a given corpus and then use it to automatically generate text.

Throughout this task we will define a single class with different functions. If you're unsure how to access class methods and attributes, take a look at the documentation (https://docs.python.org/3/tutorial/classes.html).

__a) Collecting the counts__

Write a function `process_corpus` that takes a corpus of text (as a sequence of tokens) as input and counts how often an n-gram of length $n$ (``context_len=n``) is followed by a certain word (the n-grams should __not__ be padded). The function should return a dictionary that maps every n-gram that is observed in the corpus to an inner dictionary. The inner dictionary maps each word to a number, that indicates how often the word succeeds the n-gram in the given corpus. We will need these counts to compute the conditional probabilities $P(w_{i}|w_{i-n}, ..., w_{i-1})$.
Additionally, also return the entire vocabulary $V$ (i.e. the set of all unique tokens that appear in the corpus).

Make sure your implementation is efficient by using e.g. ``Counter`` and ``defaultdict`` from the package ``collections``.   

__b) Conditional probabilities__

Write a function `transition_prob` that takes an n-gram $(w_{i-n}, ..., w_{i-1})$ and a word $w_{i}$ of the vocabulary $V$ as input and returns the conditional probability that the given n-gram is followed by the input word $w_{i}$. Recall that this conditional probability can be computed as follows:

$P(w_{i}|w_{i-n}, ..., w_{i-1}) = \frac{\text{Count}(w_{i-n}, ..., w_{i-1}, w_{i})}{\sum_{w \in V}\text{Count}(w_{i-n}, ..., w_{i-1}, w)}$

If the n-gram has never been observed, return $\frac{1}{|V|}$.

__c) Most likely word__

Write a function `most_likely_word` that gets an n-gram as input and returns the word that is most likely to succeed the given n-gram. In case there are multiple words that are equally likely to follow the given n-gram, return all of them. 
Note that you should **not** loop over the **entire** vocabulary to obtain the most likely word.
In case the given n-gram has never been observed, return the entire vocabulary.

__d) Generating text__

Write a function `generate_text` that generates a text sequence of length `k`, given a starting sequence of words (`ngram`). The function should choose the most likely next word in every step; in case there are multiple equally likely words, randomly choose one. You should return a list of ``k`` words, that includes the starting sequence and is the most probable continuation. 


Please do not implement other functions for the MarkovModel class.

Use the function signatures in the cell below.


```python
from collections import defaultdict, Counter
from nltk.util import ngrams
from typing import Sequence, Dict, Set, Tuple, List, Union
from nltk.corpus import gutenberg
from nltk.corpus.reader.util import StreamBackedCorpusView
import string
import random

class MarkovModel():
    '''Markov model for generating text.'''

    def __init__(self, tokens: Sequence[str], context_len: int):
        '''
        :param tokens: text corpus on which the model is trained on as an iterator of tokens
        :param context_len: length of the n-gram (number of preceding words)
        '''
        self.context_len = context_len
        self.counts, self.v = self.process_corpus(tokens)

    def process_corpus(self, tokens: Sequence[str]) -> tuple[Dict[Tuple[str, ...], Dict[str, int]], Set]:
        '''Training method of the model, counts the occurrences of each word after each observed n-gram.
        :param tokens: text corpus on which the model is trained on as an iterator of tokens
        :returns:
            - nested dict that maps each n-gram to the counts of the words succeeding it
            - the whole vocabulary as a set
        '''
        ngram_counts = defaultdict(Counter)
        valids = []

        # Directly perform token cleaning operations
        for token in tokens:
            token = token.lower()
            if not token.isdigit() and token not in string.punctuation:
                valids.append(token)

        vocabulary = set(valids)
        n_grams = list(ngrams(valids, self.context_len + 1))

        for ngram in n_grams:
            prefix = ngram[:-1]
            next_word = ngram[-1]
            ngram_counts[prefix][next_word] += 1

        ngram_dict = {ngram: dict(counter) for ngram, counter in ngram_counts.items()}
        return ngram_dict, vocabulary

    def transition_prob(self, ngram: Tuple[str, ...], word: str) -> float:
        '''Compute the conditional probability that the input word follows the given n-gram.
        :param ngram: string tuple that represents an n-gram
        :param word: input word
        :return: probability that the n-gram is followed by the input word
        '''
        if ngram not in self.counts:
            print("ngram not in vocabulary")
            return 1 / len(self.v)

        next_word_counts = self.counts[ngram]
        total_count = sum(next_word_counts.values())
        word_count = next_word_counts.get(word, 0)
        return word_count / total_count

    def most_likely_word(self, ngram: Tuple[str, ...]) -> Set[str]:
        '''Computes which word is most likely to follow a given n-gram.
        :param ngram: n-gram we are interested in
        return: set of words that are most likely to follow the n-gram
        '''
        if ngram not in self.counts:
            return self.v

        next_word_counts = self.counts[ngram]
        max_count = max(next_word_counts.values())
        most_likely_words = [word for word, count in next_word_counts.items() if count == max_count]
        return set(most_likely_words)

    def generate_text(self, ngram: Tuple[str, ...], k: int) -> List[str]:
        '''Generates a text sequence of length k, given a starting sequence.
        :param ngram: starting sequence
        :param k: total number of words in the generated sequence
        :return: sequence of generated words, including the starting sequence
        '''
        generated_text = list(ngram)

        current_ngram = ngram

        for _ in range(k - len(ngram)):
            most_likely_words = self.most_likely_word(current_ngram)
            next_word = random.choice(list(most_likely_words))
            generated_text.append(next_word)

            # Update the current n-gram
            current_ngram = tuple(generated_text[-self.context_len:])

        return generated_text

```

__e) Apply the model to a corpus__

Finally, we want to apply our functions to the King James Bible (`'bible-kjv.txt'`) that is part of the ``gutenberg`` corpus. Use the function from HA 1, task 2a) to obtain a list of valid tokens (do not remove stopwords) from the King Jame Bible.

Initialize the MarkovModel with the list of valid tokens and ``context_len=3`` and answer the following subtasks:

i) What is the probability that the word ``babylon`` follows the sequence ``the king of``? 

ii) What are the most likely words to follow the sequence ``the world is``? 

iii) Generate a sequence of length 20 that starts with ``mother mary was``.



```python
x= MarkovModel(gutenberg.words('bible-kjv.txt'),3)
trans_prob = x.transition_prob(('the','king','of'),'babylon')
print("Transition Probability ,",trans_prob)
most_likely_word = x.most_likely_word(('the','word','is'))
print("Most Likely word , ",most_likely_word)
generated_text = x.generate_text(('mother','mary','was'),20)
print("Generated words , ",generated_text)
```

    Transition Probability , 0.1906779661016949
    Most Likely word ,  {'sown', 'gone', 'very', 'nigh', 'not'}
    Generated words ,  ['mother', 'mary', 'was', 'espoused', 'to', 'joseph', 'before', 'they', 'came', 'together', 'she', 'was', 'found', 'with', 'child', 'of', 'gilead', 'that', 'behold', 'the']


### Task 2: POS tagging

In this task, we want to explore sentences with similar part of speech (POS) tag structure. For this, we need a corpus of text with tags. We will generate such a corpus by using NLTK’s currently recommended POS tagger to tag a given list of tokens (https://www.nltk.org/api/nltk.tag.html).


```python
# NLTK's off-the-shelf POS tagger
from nltk import pos_tag, word_tokenize
from typing import List, Dict, Set
from nltk.corpus.reader.util import StreamBackedCorpusView
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
import random
from nltk import pos_tag
nltk.download('gutenberg')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
```

    [nltk_data] Downloading package gutenberg to
    [nltk_data]     /home/parzival/nltk_data...
    [nltk_data]   Package gutenberg is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /home/parzival/nltk_data...
    [nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
    [nltk_data] Downloading package punkt to /home/parzival/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True



__a)__ Given a corpus of text ``corpus`` as a sequence of tokens, we want to collect all words that are tagged with a certain POS tag. Implement a function ``collect_words_for_tag`` that first tags the given corpus using NLTK's off-the-shelf tagger imported in the cell above. Then, for each POS tag, collect all words that were tagged with it. You should return a dictionary that maps each POS tag that was observed to the set of words that were assigned this tag in the given corpus. 


```python
from nltk.corpus.reader.util import StreamBackedCorpusView

def collect_words_for_tag(corpus: Union[List[str], StreamBackedCorpusView]) -> Dict[str, Set[str]]:
    '''
    :param corpus: sequence of tokens that represents the text corpus
    :return: dict that maps each tag to a set of tokens that were assigned this tag in the corpus
    '''

    tagged_corpus = pos_tag(corpus)  
    tag_to_words = {}
    
    for word, tag in tagged_corpus:
        if tag not in tag_to_words:
            tag_to_words[tag] = set()
        tag_to_words[tag].add(word)
    
    return tag_to_words
```

__b)__ Implement a function ``generate_sentences`` that gets a sentence and a POS dictionary (assume the POS dictionary was generated by your function in __a)__) as input and generates ``n`` sequences of words with the same tag structure. The words in your generated sequence should be randomly taken from the set of words associated with the current tag. 

Additionally, the user should have the option to achieve sentences of ``better_quality``. Thus, if ``better_quality=True``, make sure that the tag structure of the output sentences actually matches the tag structure of the input sentence, as the tags may change depending on the context. 

You can assume that the training corpus is large enough to include all possible POS tags. 

_Hint: consider the_ ``random`` _module_


```python
def generate_sentences(sentence: List[str], pos_dict: Dict[str, Set[str]], n: int, better_quality: bool = False) -> List[List[str]]:
    '''
    :param sentence: the input sentence as a list of tokens
    :param pos_dict: dictionary that maps each tag to a set of tokens
    :param n: number of sequences to generate
    :param better_quality: if True, ensures the tag structure of output sentences matches the input
    :return: list of generated sentences
    '''
    sentences = []
    tagged_sent = pos_tag(sentence)

    for _ in range(n):
        new_sentence = []
        for word, tag in tagged_sent:
            if tag in pos_dict:
                new_word = random.choice(list(pos_dict[tag]))
                new_sentence.append(new_word)
            else:
                new_sentence.append(word)
        
        if better_quality:
            new_tagged_sent = pos_tag(new_sentence)
            for i, (new_word, new_tag) in enumerate(new_tagged_sent):
                original_tag = tagged_sent[i][1]
                if new_tag != original_tag:
                    if original_tag in pos_dict:
                        new_sentence[i] = random.choice(list(pos_dict[original_tag]))

        sentences.append(new_sentence)
    
    return sentences
```

__c)__ Using the input sentence ``This test is very difficult``, test your implementation to generate 10 sentences based on  

* "Emma" by Jane Austen

* The "King James Bible"

Store your POS dictionary in ``emma_tags``and ``bible_tags``, respectively. Your generated sentences should be stored in ``emma_sent`` and ``bible_sent``. 


```python
sent = ["This", "test", "is", "very", "difficult"]
```


```python
emma_text = gutenberg.raw('austen-emma.txt')
emma_tokens = word_tokenize(emma_text)

bible_text = gutenberg.raw('bible-kjv.txt')
bible_tokens = word_tokenize(bible_text)
 
emma_tags = collect_words_for_tag(emma_tokens)
bible_tags = collect_words_for_tag(bible_tokens)

emma_sent = generate_sentences(sent, emma_tags, n=10, better_quality=True)


bible_sent = generate_sentences(sent, bible_tags, n=10, better_quality=True)

emma_sent, bible_sent
```




    ([['Those', 'delay', 'engages', 'badly', 'silver'],
      ['this', 'protestation', 'improves', 'apprehensively', 'open-hearted'],
      ['envy', 'wound', 'says', 'lovely', 'well-grown'],
      ['these', 'phrase', 'hopes', 'stoutly', 'insidious'],
      ['No', 'ballroom', 'hopes', 'just', 'mournful'],
      ['Both', 'subject.', 'tells', 'moreover', 'nourishing'],
      ['The', 'ashamed', 'Does', 'naturally', 'parish'],
      ['Some', 'analogy', 'inherits', 'deliberately', 'strict'],
      ['An', 'mortification', 'friends', 'Certainly', 'old-fashioned'],
      ['That', 'intelligence', 'arises', 'complacently', 'pink']],
     [['No', 'forehead', 'confesseth', 'Then', 'fold'],
      ['this', 'course', 'dresseth', 'heavenly', 'lighten'],
      ['the', 'fight', 'fleeth', 'once', 'drowned'],
      ['any', 'thrice', 'winds', 'dearly', 'brotherly'],
      ['these', 'excellent', 'horns', 'safely', 'oppress'],
      ['These', 'dungeon', 'contendeth', 'darkness', 'fight'],
      ['Both', 'depth', 'sorrows', 'bountifully', 'understood'],
      ['both', 'mole', 'dwell', 'brotherly', 'unicorn'],
      ['both', 'leisure', 'blindeth', 'pray', 'wellfavoured'],
      ['a', 'hospitality', 'consumeth', 'constantly', 'unripe']])



### Task 3: The Viterbi algorithm
Implement the Viterbi algorithm as introduced in the lecture (lecture 8, slide 20) and the exercise. The input of your function is a sentence that should be tagged, a dictionary with state transition probabilites and a dictionary with word emission probabilities. You may assume that the _transition probabilities_ are complete, i.e. the dictionary includes every combination of states. In contrast, we assume that all combinations of words and POS tags that are not in the dictionary of _emission probabilities_ have an emission probability of 0.

The function should return a list of POS tags, s.t. that each tag corresponds to a word of the input sentence. Moreover, return the probability of the sequence of POS tags that you found. 

You can test your function on the given example that was discussed in the Pen&Paper exercise. For the sentence ``the fans watch the race`` and the provided probabilities, your function should return the POS tag sequence ``['DT', 'N', 'V', 'DT', 'N']`` and a probability of ``9.720000000000002e-06``.

Additionally, implement beam search in the viterbi algorithm. The beam size is defined by the parameter `beam`. For example for `beam=2` we only keep the best 2 scores per column in each step and discard the rest. You may use the example from the lecture to test your implementation.


```python
# test sentence
sentence = ["the", "fans", "watch", "the", "race"]

# state transition probabilities (complete)
state_trans_prob = {('<s>','DT'):0.8,('<s>','N'):0.2,('<s>','V'):0.0,
                    ('DT','DT'):0.0,('DT','N'):0.9,('DT','V'):0.1,
                    ('N','DT'):0.0,('N','N'):0.5,('N','V'):0.5,
                    ('V','DT'):0.5,('V','N'):0.5,('V','V'):0.0}

# word emission probabilities (not complete, all combinations that are not present have probability 0)
word_emission_prob = {('the','DT'):0.2, ('fans','N'):0.1,('fans','V'):0.2,('watch','N'):0.3,
                      ('watch','V'):0.15,('race','N'):0.1,('race','V'):0.3}
```


```python
def Viterbi(sentence: List[str], trans_prob: Dict[Tuple[str,str], float], emiss_prob: Dict[Tuple[str,str], float], beam: int=0) -> tuple[List[str], float]:
    '''
    :param sentence: sentence that we want to tag
    :param trans_prob: dict with state transition probabilities
    :param emiss_prob: dict with word emission probabilities
    :param beam: beam size for beam search. If 0, don't apply beam search
    :returns:
        - list with POS tags for each input word
        - float that indicates the probability of the tag sequence
    '''
    states = {state for (_, state) in trans_prob.keys() if state != '<s>'}
    n = len(sentence)
    
    # Initialize Viterbi matrix and backpointer matrix
    V = [{} for _ in range(n)]
    B = [{} for _ in range(n)]
    
    # Initialize start probabilities
    for state in states:
        V[0][state] = trans_prob[('<s>', state)] * emiss_prob.get((sentence[0], state), 0)
        B[0][state] = None
    
    # Recursion step
    for t in range(1, n):
        for state_j in states:
            max_prob, max_state = max(
                (V[t-1][state_i] * trans_prob[(state_i, state_j)] * emiss_prob.get((sentence[t], state_j), 0), state_i)
                for state_i in states if V[t-1][state_i] > 0
            )
            V[t][state_j] = max_prob
            B[t][state_j] = max_state
        
        if beam > 0:
            # Apply beam search pruning
            sorted_states = sorted(states, key=lambda state: V[t][state], reverse=True)
            top_states = sorted_states[:beam]
            for state in states:
                if state not in top_states:
                    V[t][state] = 0
    
    # Termination step
    last_timestep = V[-1]
    max_final_state = max(last_timestep, key=last_timestep.get)
    max_final_prob = last_timestep[max_final_state]
    
    # Path backtracking
    best_path = [max_final_state]
    for t in range(n-1, 0, -1):
        best_path.insert(0, B[t][best_path[0]])
    
    return best_path,max_final_prob
```


```python
Viterbi(sentence, state_trans_prob, word_emission_prob, 2)
```




    (['DT', 'N', 'V', 'DT', 'N'], 9.720000000000002e-06)


