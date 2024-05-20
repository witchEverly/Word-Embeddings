# Word-Embeddings

## One-hot Encoding

Our goal is to find a numeric representation of a word. Now the question is how to represent a word as a number. The most basic way we can do that is with a one-hot encoding.

$$
|V| = 10,000
$$

Note: $|V|$ is the size of our vocabulary; we use brackets to denote cardinality.

Out of 10,000 words, we can represent each word as a vector of 10,000 dimensions. Each word is represented by a vector of 10,000 dimensions.

So if word $w$ is at index $i$, then the vector representation of $w$ is a vector of 10,000 dimensions with all zeros except at index $i$ where it is 1.

$$
w_0 = [1, 0, 0, \ldots, 0]^T
$$

$$
w_1 = [0, 1, 0, \ldots, 0]^T
$$

$$
.
.
.
$$

$$
w_{9999} = [0, 0, 0, \ldots, 1]^T
$$

The matrix representing our vocabulary is called the embedding matrix, and it looks like this:

$$
\begin{equation}
    \text{embedding matrix}_{|V| \times |V|} =
        \begin{bmatrix}
            1 & 0 & 0 & \ldots & 0 \cr
            0 & 1 & 0 & \ldots & 0 \cr
            \vdots & \vdots & \vdots & \ddots & \vdots \cr
            0 & 0 & 0 & \ldots & 1
\end{bmatrix}
\end{equation}
$$

## The Problem with One-hot Encoding

You may think this is a very inefficient way to represent words due to the sparsity and the high dimensionality of the vectors. A very large matrix would be very slow to train and very slow to run when it comes to a neural network.

**But that isn't the only reason why one-hot encoding is inefficient.** The main reason is that one-hot encoding doesn't capture the similarity between words. This is because the Euclidean distance between any two one-hot vectors is the same, and the cosine similarity between any two one-hot vectors is 0.

Let's demonstrate this mathematically:

The Euclidean distance between two vectors $v_i$ and $v_j$ is defined as:

$$
d(v_i, v_j) = \sqrt{\sum_{k=1}^{n} (v_{ik} - v_{jk})^2}
$$

where $v_i$ and $v_j$ are the one-hot vectors of words $w_i$ and $w_j$.

Let's calculate the Euclidean distance between two one-hot vectors $w_0$ and $w_1$:

$$
d(w_0, w_1) = \sqrt{(1-0)^2 + (0-1)^2 + 0 + \ldots + 0} = \sqrt{2}
$$

Similarly, the distance between $w_0$ and $w_{9999}$ is:

$$
d(w_0, w_{9999}) = \sqrt{(1-0)^2 + 0 + \ldots + (0-1)^2} = \sqrt{2}
$$

We have shown that the Euclidean distance between any two one-hot vectors is $\sqrt{2}$. Which isn't very helpful for determining the distance between words.

Now let's calculate the cosine similarity between two one-hot vectors $w_0$ and $w_1$:

$$
cos(v_i, v_j) = \frac{v_i \cdot v_j}{||v_i|| \cdot ||v_j||}
$$

again, where $v_i$ and $v_j$ are the one-hot vectors of words $w_i$ and $w_j$.

$$
cos(w_0, w_1) = \frac{(1 \cdot 0) + (0 \cdot 1) + 0 + \ldots + 0}{\sqrt{1} \cdot \sqrt{1}} = 0
$$

and for $w_0$ and $w_{9999}$ the cosine similarity is:

$$
cos(w_0, w_{9999}) = \frac{(1 \cdot 0) + 0 + \ldots + (0 \cdot 1)}{\sqrt{1} \cdot \sqrt{1}} = 0
$$


We have shown that the cosine similarity between any two one-hot vectors is 0. Which isn't very helpful for determining the similarity between words.


## Feature Representation

The solution to this problem is to represent words as dense vectors. This is where word embeddings come in. The basic idea behind word embeddings is to represent words in a continuous vector space. This is done by representing each word as a dense vector of a fixed dimensionality. For each dimension, we learn a real-valued number that represents a feature of the word.

This is a dummy example of a word embedding matrix with 3 words and 4 features:

$$
\begin{equation}
    \begin{array}{c|cccc}
        & \text{aquatic} & \text{mammal} & \text{size} & \text{color} \\
        \hline
        \text{cat} & 0.1 & 0.9 & 0.4 & 0.4 \\
        \text{dog} & 0.2 & 0.8 & 0.7 & 0.8 \\
        \text{fish} & 0.9 & 0.5 & 0.2 & 0.9
    \end{array}
\end{equation}
$$
