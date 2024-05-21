
# Neural Probabilistic Language Model (NPLM)

- Introduced by Bengio et al. in 2003.
- Based on a feedforward neural network with a single hidden layer. 
- The input to the network is a one-hot encoded vector of the current
- The output is a probability distribution over the vocabulary. 
- The hidden layer is a non-linear function of the input word
- The weights connecting the input layer to the hidden layer are shared across all words. 
- The weights connecting the hidden layer to the output layer are not shared.

> Paper: 
> [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## High-Level Architecture: NPLM

The high-level architecture of NPLM is as follows:

1. **Input Layer**: The input layer is a one-hot encoded vector of the current word. The input layer has a dimensionality equal to the size of the vocabulary.
2. **Hidden Layer**: The hidden layer is a non-linear function of the input word. The weights connecting the input layer to the hidden layer are shared across all words. The hidden layer has a dimensionality equal to the size of the hidden layer.
3. **Output Layer**: The output layer is a probability distribution over the vocabulary. The weights connecting the hidden layer to the output layer are not shared. The output layer has a dimensionality equal to the size of the vocabulary.
4. **Training**: The model is trained using the maximum likelihood estimation (MLE) criterion. The objective is to maximize the log-likelihood of the training data.
5. **Inference**: During inference, the model is used to predict the next word in a sequence given the previous words.
6. **Word Embeddings**: The weights connecting the input layer to the hidden layer are the word embeddings. The word embeddings are learned during training.
7. **Shared Weights**: The weights connecting the input layer to the hidden layer are shared across all words. This allows the model to generalize to unseen words.
8. **Non-linear Activation Function**: The hidden layer uses a non-linear activation function such as the hyperbolic tangent (tanh) or rectified linear unit (ReLU) function. This allows the model to capture complex patterns in the data. The paper uses the hyperbolic tangent (tanh) activation function.
9. **Softmax Activation Function**: The output layer uses a softmax activation function to produce a probability distribution over the vocabulary. The model is trained to maximize the log-likelihood of the training data.

### Model Architecture: NPLM in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NPLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NPLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.tanh(self.hidden(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)
```


## Pre-training and Fine-tuning

NPLM can be pre-trained on a large corpus of text using unsupervised learning. The pre-trained model can then be fine-tuned on a smaller corpus of text using supervised learning. This allows the model to learn the specific characteristics of the smaller corpus while retaining the general knowledge learned from the larger corpus.

### Key Features of NPLM

# Word2Vec

> Papers: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
> 
> [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

- Shallow, two-layer neural network model for learning word embeddings.
- Trained to reconstruct linguistic contexts of words.
- Takes a large corpus of text as input and produces a vector space of word embeddings.
- Words that share common contexts are located close to each other in the vector space.
- Two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.
  - CBOW predicts the current word given the context words.
  - Skip-gram predicts the context words given the current word.

## High-Level Architecture: Word2Vec

The high-level architecture of Word2Vec is as follows:

1. **Input Layer**: The input layer is a one-hot encoded vector of the current word. The input layer has a dimensionality equal to the size of the vocabulary.
2. **Hidden Layer**: The hidden layer is a non-linear function of the input word. The weights connecting the input layer to the hidden layer are shared across all words. The hidden layer has a dimensionality equal to the size of the hidden layer.
3. **Output Layer**: The output layer is a probability distribution over the vocabulary. The weights connecting the hidden layer to the output layer are not shared. The output layer has a dimensionality equal to the size of the vocabulary.
4. **Training**: The model is trained using the maximum likelihood estimation (MLE) criterion. The objective is to maximize the log-likelihood of the training data.
5. **Inference**: During inference, the model is used to predict the next word in a sequence given the previous words.
6. **Word Embeddings**: The weights connecting the input layer to the hidden layer are the word embeddings. The word embeddings are learned during training.
7. **Shared Weights**: The weights connecting the input layer to the hidden layer are shared across all words. This allows the model to generalize to unseen words.
8. **Non-linear Activation Function**: The hidden layer uses a non-linear activation function such as the hyperbolic tangent (tanh) or rectified linear unit (ReLU) function. This allows the model to capture complex patterns in the data. The paper uses the hyperbolic tangent (tanh) activation function.
9. **Softmax Activation Function**: The output layer uses a softmax activation function to produce a probability distribution over the vocabulary. The model is trained to maximize the log-likelihood of the training data.
10. **Two Architectures**: Word2Vec has two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.
    - **Continuous Bag of Words (CBOW)**: CBOW predicts the current word given the context words.
    - **Skip-gram**: Skip-gram predicts the context words given the current word.

### Model Architecture: Word2Vec in PyTorch (CBOW)

```python
import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.output(x)
        return x
```


### Model Architecture: Word2Vec in PyTorch (Skip-gram)

```python
import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.output(x)
        return x
```




### Key Features of Word2Vec

- Distributed Representations: Word2Vec learns distributed representations of words in a continuous vector space.
- Contextual Similarity: Words that share common contexts are located close to each other in the vector space.
- Two Architectures: Word2Vec has two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.

# Doc2Vec

> Paper: [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

Doc2Vec is an extension of Word2Vec that learns to correlate labels with words. It is a shallow, two-layer neural network that is trained to predict the next word in a sentence given the previous words and a unique label for the sentence. The unique label is a tag that is added to all words in a sentence. The model is trained to predict the next word in a sentence given the previous words and the unique label. The weights connecting the input layer to the hidden layer are shared across all words in the sentence, but the weights connecting the hidden layer to the output layer are not shared.

# FastText

> Paper: [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)

FastText is a library for learning of word embeddings and text classification created by Facebook's AI Research lab. The model is an extension of Word2Vec that is able to capture **subword information**. FastText represents each word as a bag of character n-grams. The vector for a word is the sum of the vectors of its character n-grams. This allows FastText to capture morphological information about words.

### Key Features of FastText

- Subword Information: FastText represents each word as a bag of character n-grams. This allows FastText to capture morphological information about words.

# GloVe

> Paper: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words. The model is based on matrix factorization techniques on the word-word co-occurrence matrix. The resulting word vectors are learned such that their dot product equals the logarithm of the words' probability of co-occurrence.

### Key Features of GloVe

- Global Statistics: GloVe is based on the global statistics of the corpus. It uses the word-word co-occurrence matrix to learn the word vectors.
- Log-bilinear Model: GloVe is based on a log-bilinear model that learns word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence.

# ELMo

> Paper: [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

ELMo (Embeddings from Language Models) is a deep contextualized word representation model. The model is based on a bidirectional LSTM language model. ELMo takes as input a sequence of words and produces a set of word vectors as output. The word vectors are contextualized embeddings that capture the meaning of the word in the context of the sentence.

### Key Features of ELMo

- Contextualized Embeddings: ELMo produces contextualized word embeddings that capture the meaning of the word in the context of the sentence.
- Bidirectional LSTM: ELMo is based on a bidirectional LSTM language model that captures the context of the word in both directions.
