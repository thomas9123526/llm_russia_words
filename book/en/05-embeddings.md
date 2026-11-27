# Chapter 5. Embeddings: Words as Vectors (Word2Vec, GloVe)

> **Part 1 — Text and Representation** · Level: Beginner
>
> **In this chapter you will learn:** why words should be represented by dense vectors · how the meaning of a word can be learned from its neighbours (Word2Vec, GloVe) · what embeddings capture well and where they fail

## How can a model know that two words are similar?

At the end of Chapter 4 we found a basic weakness of counting: for an n-gram model, *cat* and *kitten* are as unrelated as *cat* and *inflation*. The one-hot vectors of Chapter 2 have the same problem. The **dot product** of two vectors $\mathbf{a}$ and $\mathbf{b}$ is $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$, and for two different one-hot vectors it is always 0. Every word is equally far from every other word.

We would like something better: a representation in which each word is a point in a space, and words with similar meanings are close to each other. Then what a model learns about *cat* will partly transfer to *kitten*. Such a representation is called a **word embedding**: each word is mapped ("embedded") to a short vector of real numbers, for example of length 100 or 300, instead of a one-hot vector of length 50,000. Unlike one-hot vectors, these vectors are **dense**: most of their entries are non-zero.

But where do these numbers come from? Nobody can write them by hand. This chapter shows how to learn them from raw text.

## You shall know a word by the company it keeps

Here is a small experiment. You probably do not know the word *tezgüino*. Now read:

- *A bottle of tezgüino is on the table.*
- *Everybody likes tezgüino.*
- *Tezgüino makes you drunk.*
- *We make tezgüino out of corn.*

After four sentences you can guess that *tezgüino* is some kind of alcoholic drink, similar to beer. You learned this only from the words around it. This idea is the **distributional hypothesis**: words that occur in similar contexts tend to have similar meanings. It was formulated by the linguist Zellig Harris in the 1950s, and John Firth summarized it in a famous phrase: *"You shall know a word by the company it keeps."*

The hypothesis gives us a recipe. If we describe each word by the contexts it appears in, words with similar meanings will get similar descriptions. The simplest way is to count. Choose a **context window** — for example, one or two words to the left and right — and build a **co-occurrence matrix** $X$ of size $V \times V$, where $X_{ij}$ is the number of times word $j$ appears in the window around word $i$. Each row of $X$ is a description of one word. If *tea* and *coffee* both appear after *drink* and *like*, their rows will be similar.

Raw co-occurrence rows are long (length $V$) and sparse, and they are dominated by frequent words such as *the*. Classic methods therefore reweight the counts — for example with **pointwise mutual information** (PMI), which, like TF-IDF in Chapter 4, rewards pairs that occur together more often than chance — and then compress the matrix with a **matrix factorization** method such as the singular value decomposition (SVD). The result is a short dense vector for each word. This approach, known as latent semantic analysis, worked surprisingly well. Word2Vec and GloVe, which we study next, can be seen as more efficient and more accurate ways to do the same thing.

## Measuring similarity: the cosine

Once words are vectors, we need a way to say how similar two vectors are. The standard choice is **cosine similarity**, the cosine of the angle between them:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert},$$

where $\lVert \mathbf{a} \rVert = \sqrt{\sum_i a_i^2}$ is the length of $\mathbf{a}$. The value is 1 when the vectors point in the same direction, 0 when they are perpendicular, and $-1$ when they point in opposite directions. A small example: for $\mathbf{a} = (1, 2)$ and $\mathbf{b} = (2, 3)$ we get $\mathbf{a} \cdot \mathbf{b} = 8$, $\lVert \mathbf{a} \rVert = \sqrt{5}$, $\lVert \mathbf{b} \rVert = \sqrt{13}$, so $\cos = 8 / \sqrt{65} \approx 0.99$. The cosine ignores the lengths of the vectors and looks only at their directions; this is useful because the length of a word vector often reflects word frequency rather than meaning.

All embeddings of a vocabulary are stored together in an **embedding matrix** $E$ of size $V \times d$, where $d$ is the embedding dimension. Row $i$ of $E$ is the vector of word $i$. Getting the embedding of a token is just looking up a row by its index (Chapter 2). Mathematically, this is the same as multiplying the one-hot vector of the token by $E$, but no real multiplication is needed. This lookup table is exactly the first layer of every modern language model; in PyTorch it is called `nn.Embedding` (Chapter 20).

## Word2Vec

In 2013, Mikolov and colleagues at Google introduced **Word2Vec**, a family of fast methods that learn embeddings by *prediction* instead of counting. There are two variants:

- **Skip-gram**: given a word, predict the words in its context window.
- **CBOW** (continuous bag of words): given the context words, predict the word in the middle.

Let us look at skip-gram. Each word $w$ has two vectors: a "target" vector $\mathbf{v}_w$ (used when $w$ is the centre word) and a "context" vector $\mathbf{u}_w$ (used when $w$ appears in the context). The model turns their dot product into a probability with the **softmax** function:

$$P(c \mid w) = \frac{\exp(\mathbf{u}_c \cdot \mathbf{v}_w)}{\sum_{c'=1}^{V} \exp(\mathbf{u}_{c'} \cdot \mathbf{v}_w)}.$$

The numerator is large when $\mathbf{u}_c$ and $\mathbf{v}_w$ point in similar directions; the denominator, a sum over the whole vocabulary, makes the probabilities add up to 1. Training adjusts all vectors so that real context words get high probability.

The problem is the denominator: computing it for every training pair means a sum over tens of thousands of words. Word2Vec avoids this with **negative sampling**. Instead of a full softmax, the model solves a simpler yes/no task: is the pair $(w, c)$ a real pair from the text, or a random one? For each real pair, we draw $k$ random "negative" words $n_1, \dots, n_k$ and minimize

$$L = -\log \sigma(\mathbf{u}_c \cdot \mathbf{v}_w) - \sum_{j=1}^{k} \log \sigma(-\mathbf{u}_{n_j} \cdot \mathbf{v}_w),$$

where $\sigma(x) = 1 / (1 + e^{-x})$ is the **sigmoid** function, which squeezes any number into the interval $(0, 1)$. The first term pushes the real pair's score up; the second pushes the random pairs' scores down. In the original method negative words are drawn with probability proportional to their frequency raised to the power $3/4$, and $k$ is between about 2 and 20.

Here is a complete skip-gram with negative sampling in NumPy, trained on a toy corpus. It uses gradient descent, which we will study properly in Chapter 8. For now it is enough to know that each update moves the vectors a little in the direction that decreases $L$. The derivative of $L$ with respect to each score $s = \mathbf{u} \cdot \mathbf{v}$ is simply $\sigma(s) - y$, where $y = 1$ for the real pair and $y = 0$ for the negatives.

```python
import numpy as np

corpus = ["i like tea", "i like coffee", "i drink tea", "i drink coffee",
          "dogs chase cats", "cats chase mice", "dogs chase mice"]
words = sorted({w for s in corpus for w in s.split()})
idx = {w: i for i, w in enumerate(words)}
rng = np.random.default_rng(0)
V, d, k, lr = len(words), 5, 3, 0.05   # vocabulary size, dimension, negatives, step
W_in = rng.normal(0, 0.1, (V, d))      # target-word vectors (the embeddings)
W_out = rng.normal(0, 0.1, (V, d))     # context-word vectors
sigmoid = lambda x: 1 / (1 + np.exp(-x))

for epoch in range(300):
    for t in (s.split() for s in corpus):
        for i, w in enumerate(t):
            for j in range(max(0, i - 1), min(len(t), i + 2)):   # window of 1
                if j == i:
                    continue
                # 1 real context word + k random "negative" words (uniform here)
                ids = np.array([idx[t[j]]] + list(rng.integers(0, V, k)))
                labels = np.array([1.0] + [0.0] * k)
                v, u = W_in[idx[w]], W_out[ids]
                g = sigmoid(u @ v) - labels        # gradient of the loss w.r.t. scores
                W_in[idx[w]] -= lr * g @ u
                np.add.at(W_out, ids, -lr * np.outer(g, v))

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
for a, b in [("tea", "coffee"), ("like", "drink"), ("tea", "dogs")]:
    print(a, b, round(float(cosine(W_in[idx[a]], W_in[idx[b]])), 2))
```

The program prints `tea coffee 1.0`, `like drink 1.0` and `tea dogs 0.49`. The words *tea* and *coffee* never appear together, but they appear in the same contexts, so their vectors become almost identical; the same holds for *like* and *drink*. The unrelated pair *tea*–*dogs* is much less similar. (The exact numbers depend on the random seed; with a real corpus, unrelated words usually have cosines close to 0.) We use `np.add.at` because the same word may appear twice among the sampled ids.

## GloVe

**GloVe** (Global Vectors), proposed by Pennington, Socher and Manning in 2014, connects the two approaches. It first builds the global co-occurrence matrix $X$, like the count-based methods, and then learns vectors whose dot products predict the logarithms of the counts:

$$J = \sum_{i,j} f(X_{ij}) \left( \mathbf{w}_i \cdot \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2.$$

Here $\mathbf{w}_i$ is the vector of word $i$, $\tilde{\mathbf{w}}_j$ the context vector of word $j$, $b_i$ and $\tilde{b}_j$ are scalar biases, and the sum runs over pairs with $X_{ij} > 0$. The weighting function $f$ gives small weight to rare pairs (whose counts are noisy) and limits the weight of very frequent pairs. In practice, Word2Vec and GloVe give embeddings of similar quality; it was later shown that skip-gram with negative sampling is itself implicitly factorizing a shifted PMI matrix, so the two families are closer than they look.

## What embeddings capture — and what they miss

Trained on large corpora, word embeddings show striking regularities. The most famous is **word analogy** by **vector arithmetic**:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}.$$

The difference $\mathbf{v}_{\text{woman}} - \mathbf{v}_{\text{man}}$ acts like a "gender direction"; similar directions appear for country–capital (*Франция → Париж, Россия → Москва*) and for grammar (*идти → шёл*). These analogies are not perfect, and they are found by searching for the nearest word to the computed vector, but they show that the space has meaningful structure.

Classic word embeddings also have clear limitations:

1. **Unknown and rare words.** Each word form gets its own vector, so a form never seen in training has no vector. This hurts Russian, where one word has many forms. **fastText** (Bojanowski et al., 2017) represents a word as the sum of vectors of its character n-grams, so *котами* shares pieces with *кот* and *котом*, and even unseen forms get vectors — the same idea as subword tokenization in Chapter 3.
2. **One vector per word.** These embeddings are **static**: a word has the same vector in every sentence. But the Russian *ключ* can be a key or a spring of water, and the Korean *배* can be a pear, a ship or a belly. A static vector mixes all meanings together.
3. **Bias.** Embeddings learn whatever regularities exist in the text, including social stereotypes, for example associations between professions and gender. We return to this in Chapter 40.

The second limitation leads to the main idea of modern language models. Instead of one fixed vector per word, a Transformer starts from a static embedding (a row of $E$) and then computes a **contextual representation** of each token that depends on the whole sentence, so *ключ* in *«ключ от двери»* and in *«горный ключ»* gets different vectors. To understand how such networks are built and trained, we first need the tools of Part 2: vectors and matrices, layers, loss functions and gradient descent.

## Summary

- One-hot vectors make all words equally different; word embeddings map each word to a short dense vector where similar words are close.
- The distributional hypothesis — words in similar contexts have similar meanings — lets us learn embeddings from raw text, either by counting co-occurrences and factorizing the matrix or by prediction.
- Cosine similarity measures how close two vectors are by the angle between them; all embeddings form an embedding matrix, and lookup is row selection.
- Word2Vec (skip-gram, CBOW) learns vectors by predicting context words, made efficient by negative sampling; GloVe fits dot products to logarithms of global co-occurrence counts.
- Embeddings capture analogies, but static word vectors cannot handle unseen forms (fastText helps), mix the senses of ambiguous words, and absorb biases; contextual representations in Transformers address the second problem.

## Exercises

1. Explain in your own words the distributional hypothesis, and give your own example in Russian or Korean of a word whose meaning you can guess from context.
2. Compute the cosine similarity of $\mathbf{a} = (1, 0, 2)$ and $\mathbf{b} = (2, 1, 1)$ by hand, then check with NumPy.
3. In the skip-gram code, change the dimension `d` to 2 and the seed to 1, 2, 3. How stable are the similarities? Then add the sentences *"i like juice"* and *"i drink juice"*: which word becomes closest to *juice*?
4. Why can a static embedding not distinguish the two meanings of *ключ*? Describe in two or three sentences what information a model would need to tell them apart.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| word embedding | векторное представление слова | 단어 임베딩 |
| dense vector | плотный вектор | 밀집 벡터 |
| distributional hypothesis | дистрибутивная гипотеза | 분포 가설 |
| context window | контекстное окно | 문맥 창 |
| co-occurrence matrix | матрица совместной встречаемости | 동시 출현 행렬 |
| pointwise mutual information (PMI) | поточечная взаимная информация | 점별 상호 정보량(PMI) |
| dot product | скалярное произведение | 내적 |
| cosine similarity | косинусное сходство | 코사인 유사도 |
| embedding matrix | матрица векторных представлений | 임베딩 행렬 |
| skip-gram | skip-gram | 스킵그램 |
| negative sampling | отрицательное сэмплирование | 네거티브 샘플링 |
| softmax | функция softmax | 소프트맥스 |
| sigmoid | сигмоида | 시그모이드 |
| word analogy | аналогия слов | 단어 유추 |
| contextual representation | контекстное представление | 문맥적 표현 |

---
[← Previous: Counting Words: Bag of Words, N-grams and Perplexity](04-counting-words.md) · [Contents](README.md) · [Next: The Math You Need: Vectors, Matrices, Probability →](06-math-you-need.md)
