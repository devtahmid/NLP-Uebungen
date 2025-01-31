# Overview of the Task (the actual notebook is present within this same folder)

### Task 1
Just getting familiar with regex in python

### Task 2: Finding the most similar word

The goal of this task is, given a corpus, to find the most similar word for a provided word. As an example we will consider the King James Bible that is is included in the `gutenberg` corpus and we are looking to find the word that is most similar to `god`. We consider two words similar if they appear in the same word context.

**a) Cleaning the input**

**b) Counting the surroundings**

**c) Keeping the top kkk words in context**

**d) Finding the most similar words**

**e) Bringing it all together**

Finally, we want to apply our functions to the King James Bible (`'bible-kjv.txt'`) that is part of the `gutenberg` corpus. We intend to find the word(s) that is (are) most similar to `god`. Follow the steps below:

i) Obtain a list of all tokens from the King James Bible and store it in a variable `tokens`.

ii) Clean the list of tokens with your function from **a)** to get the list of valid tokens (without removing stopwords) and store it in a variable `valid_tokens`.

iii) Apply your function from **b)** to count the context words for all valid tokens with a `context_size` of 2 and store the result in a variable `context_counts`.

iv) Using your function from **c)**, keep only the 20 most frequent words in a valid tokens context and store the result in a variable `context_words`.

v) Finally, find the most similar words to the word _god_ with your function from **d)** and store the set of most similar words in the variable `set_god` and the highest Jaccard similarity in the variable `sim_god`.
