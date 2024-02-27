from collections import Counter
import numpy as np
import math

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"

# UTILITY FUNCTIONS
def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """

  n_grams = []
  
  for i in range(len(tokens) - n + 1):
    n_gram = tuple(tokens[i:i + n])
    n_grams.append(n_gram)

  return n_grams

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens

def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total

class LanguageModel:

  def __init__(self, n_gram):
    """
    Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    
    self.n_gram = n_gram
    self.n_grams_count = Counter()  # number of n grams in our model
    self.n_1_grams_count = Counter() # number of n - 1 grams in our model
    self.total_n_grams = 0 # store count of n grams
    self.total_n_1_grams = 0 # store coutn of n-1 grams
    self.vocab = Counter()
    self.probabilities = {}

  def train(self, tokens: list, verbose: bool = False) -> None:
    """
    Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    
    n = self.n_gram
    
    token_count = Counter(tokens)
    
    # Determine singleton tokens
    singletons = {token for token, count in token_count.items() if count == 1}

    # Replace tokens occurring only once with <UNK>
    tokens = [token if token not in singletons else UNK for token in tokens]

    ## Check how to use Counter() for faster implementation
    n_grams = create_ngrams(tokens, n)
    self.n_grams_count.update(n_grams)
    self.total_n_grams += len(n_grams)

    if n > 1: # to get the n-1 grams 
      n_garms_1 = create_ngrams(tokens, n-1)
      self.n_1_grams_count.update(n_garms_1)
      self.total_n_1_grams += len(n_garms_1)

    self.vocab.update(tokens)

    for n_gram in self.n_grams_count:
      self.probabilities[n_gram] = self.laplace_smoothing(len(tokens), n_gram)
  
  def replace_unknown_tokens(self, sentence_tokens: list) -> list:
    """
    Replaces tokens in a list with <UNK> if they are not present in the vocabulary or occur only once.
    Args:
        sentence_tokens (list): A list of tokens to be checked and replaced if necessary.
    Returns:
        list: A new list of tokens where tokens not in the vocabulary or occurring only once are replaced with <UNK>.
    """
    updated_tokens = []
    
    for token in sentence_tokens:    
        if (token in self.vocab and self.vocab[token] > 1) or (token in {SENTENCE_BEGIN, SENTENCE_END}):
            updated_tokens.append(token)
        else:
            updated_tokens.append(UNK)
        
    return updated_tokens

  def score(self, sentence_tokens: list) -> float:
    """
    Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
    Returns:
      float: the probability value of the given tokens for this model
    """    
    # Replace tokens which are not in vocab or occurring only once in vocab with <UNK>
    sentence_tokens = self.replace_unknown_tokens(sentence_tokens)

    n_grams = create_ngrams(sentence_tokens, self.n_gram)

    probability = 1.0
    for n_gram in n_grams:
      try:
        probability *= self.probabilities[n_gram]
      except:
        probability *= self.laplace_smoothing(len(sentence_tokens), n_gram)
    
    return probability

  def laplace_smoothing(self, n: int, token: tuple) -> float:
      """
      Adds one(k-value) to all the n-gram counts and normalizes to probabilities.
      Args:
        n : the number of tokens
        token : the n-gram
      Returns:
        float: the add one smoothed probability of the n-gram
      """
  
      if self.n_gram == 1:
        return (self.n_grams_count.get(token, 0) + 1) / (n + len(self.vocab))
      else:
        n_1_gram = tuple(token[0:len(token) - 1]) # n-1 gram
        return (self.n_grams_count.get(token, 0) + 1) / (self.n_1_grams_count.get(n_1_gram, 0) + len(self.vocab))

  def generate_next_token(self, context: tuple) -> str:
    """
    Generates the next token given a context.
    Args:
      context (tuple): a tuple representing the context of the next token
    Returns:
      str: the generated next token
    """
    # Get all possible n-grams with the given context
    possible_ngrams = [ngram for ngram in self.n_grams_count.keys() if ngram[:-1] == context]

    # If no n-grams found with the given context, return sentence end token
    if len(possible_ngrams) == 0:
      return SENTENCE_END

    # Calculate probabilities of possible next tokens
    probabilities = []
    total_probability = 0.0
    for ngram in possible_ngrams:
      probability = (self.n_grams_count[ngram] + 1) / (self.total_n_grams + len(self.vocab))
      total_probability += probability
      probabilities.append((ngram[-1], probability))

    # Normalize probabilities
    normalized_probabilities = [(token_prob[0], token_prob[1] / total_probability) for token_prob in probabilities]

    # print(normalized_probabilities)

    # Choose the next token based on probabilities
    next_token = np.random.choice([token_prob[0] for token_prob in normalized_probabilities], p=[token_prob[1] for token_prob in normalized_probabilities])

    return next_token

  def generate_sentence(self) -> list:
    """
    Generates a single sentence from a trained language model using the Shannon technique.  
    Returns:
      list: the generated sentence as a list of tokens
    """
    # Initialize sentence with sentence begin token(s)
    sentence = [SENTENCE_BEGIN] * (self.n_gram - 1)

    # Generate tokens until sentence end token is encountered
    while True:
      # Generate next token
      if self.n_gram == 1:
        context = tuple()
      else:
        context = tuple(sentence[-(self.n_gram - 1):])
      next_token = self.generate_next_token(context)

      # Break if sentence end token is generated
      if next_token == SENTENCE_END or next_token == SENTENCE_BEGIN:
        sentence.append(SENTENCE_END)  
        break

      # Append token to sentence
      sentence.append(next_token)

    return sentence

  def generate(self, n: int) -> list:
    """
    Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate  
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    
    return [self.generate_sentence() for i in range(n)]

  def perplexity(self, sequence: list) -> float:
    """
    Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model  
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    # removing the <s> token from the sequence
    if self.n_gram == 1:
      sequence = [token for token in sequence if token != SENTENCE_BEGIN]
      N = len(sequence)
    
    else:
      sequence_cpy = [token for token in sequence if token != SENTENCE_BEGIN]
      N = len(sequence_cpy)

    score = self.score(sequence)

    return score ** (-1 / N)

if __name__ == '__main__':

  # init language model
  n_val = 5
  model = LanguageModel(n_val)

  # train
  train_data = read_file(r'training_files/berp-training.txt')
  tokens = tokenize(train_data, n_val, False)
  model.train(tokens)

  # test
  test_data = read_file(r'testing_files/berp-test.txt')
  test_tokens = tokenize(test_data, n_val, False)
  test_tokens = test_tokens[0:100]

  print("Score:")
  print(model.score(tokenize_line("apples are fruit", 2, False)))

  print("Perplexity:")
  print(model.perplexity(test_tokens))

  print("Generated Sentences:")
  sents = model.generate(10)
  for i, sentence in enumerate(sents, start=1):
    formatted_sentence = ' '.join(sentence).replace('<s>', '').replace('</s>', '').strip()
    print(f"Sentence {i}: {formatted_sentence}")