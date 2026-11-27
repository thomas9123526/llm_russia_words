# Chapter 6. The Math You Need: Vectors, Matrices, Probability

> **Part 2 — Neural Network Foundations** · Level: Beginner
>
> **In this chapter you will learn:** how vectors and the dot product measure similarity · how a matrix transforms one vector into another · how probability, logarithms and derivatives prepare us for training neural networks

## A question to start with

In Chapter 5 we represented the words *cat* and *dog* as vectors, and we said that similar words have "close" vectors. But what does "close" mean for a list of numbers? And when a language model predicts the next word, how does it turn a list of raw scores into probabilities that add up to one?

These two questions contain almost all the mathematics of this book. A neural network is mostly three things: vectors (lists of numbers), matrices (tables of numbers that transform vectors) and probabilities (numbers between 0 and 1 that describe uncertainty). To *train* the network we also need one idea from calculus: the derivative. This chapter introduces each of them with small examples. You do not need to be a mathematician. You need to be comfortable reading the notation, because every later chapter uses it.

## Vectors and the dot product

A **vector** is an ordered list of numbers. We write vectors as bold lowercase letters, for example

$$\mathbf{a} = (0.9,\ 0.1,\ 0.4).$$

The numbers are the **components** of the vector, and their count is its **dimension**. This vector has dimension 3; we write $\mathbf{a} \in \mathbb{R}^3$, which reads "$\mathbf{a}$ is a vector of three real numbers". Real embeddings have hundreds or thousands of dimensions, but everything works the same way as in three.

A vector can be seen in two ways. Geometrically, it is an arrow from the origin to a point in space. As data, it is a description of an object: each component measures one feature. Both views are useful.

The most important operation on two vectors is the **dot product**. Multiply the components pairwise and add the results:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n.$$

Here $n$ is the dimension, and $a_i$, $b_i$ are the $i$-th components. Let $\mathbf{a}$ be the vector for *cat* and $\mathbf{b} = (0.8,\ 0.2,\ 0.5)$ the vector for *dog*. Then

$$\mathbf{a} \cdot \mathbf{b} = 0.9 \cdot 0.8 + 0.1 \cdot 0.2 + 0.4 \cdot 0.5 = 0.72 + 0.02 + 0.20 = 0.94.$$

The dot product is large when the two vectors point in similar directions and their components are large. It is zero when they are perpendicular, and negative when they point in roughly opposite directions.

The length of a vector is called its **norm**. The usual (Euclidean) norm is

$$\|\mathbf{a}\| = \sqrt{\mathbf{a} \cdot \mathbf{a}} = \sqrt{a_1^2 + \dots + a_n^2}.$$

If we divide the dot product by both norms, length no longer matters and only direction remains. This is the **cosine similarity** that we used in Chapter 5:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}.$$

For our example, $\|\mathbf{a}\| = \sqrt{0.98} \approx 0.990$ and $\|\mathbf{b}\| = \sqrt{0.93} \approx 0.964$, so the cosine is $0.94 / 0.955 \approx 0.985$. The value is close to 1: the two arrows point almost the same way. Remember the dot product well. In Chapter 15 we will see that attention is, at its core, a dot product between a "query" vector and "key" vectors.

## Matrices: machines that transform vectors

A **matrix** is a rectangular table of numbers. A matrix with $m$ rows and $n$ columns has **shape** $m \times n$; we write $W \in \mathbb{R}^{m \times n}$. The entry in row $i$ and column $j$ is $W_{ij}$.

The key operation is **matrix–vector multiplication**. Each row of the matrix takes a dot product with the vector:

$$(W\mathbf{x})_i = \sum_{j=1}^{n} W_{ij}\, x_j.$$

So a matrix of shape $m \times n$ turns a vector of dimension $n$ into a vector of dimension $m$. Think of it as a machine: a vector goes in, a new vector comes out. For example, with

$$W = \begin{pmatrix} 1 & 0 & -1 \\ 0.5 & 0.5 & 0.5 \end{pmatrix}, \qquad W\mathbf{a} = \begin{pmatrix} 0.9 - 0.4 \\ 0.5\,(0.9 + 0.1 + 0.4) \end{pmatrix} = \begin{pmatrix} 0.5 \\ 0.7 \end{pmatrix}.$$

The first row measures "how much more of feature 1 than feature 3"; the second row measures the average size of all features, scaled. Every row is a small detector, and the output vector lists what each detector found. This is exactly what a layer of a neural network does, as we will see in Chapter 7.

**Matrix multiplication** $C = AB$ applies the same rule to many columns at once: $C_{ij} = \sum_k A_{ik} B_{kj}$. It is defined only when the inner shapes agree: $(m \times k)$ times $(k \times n)$ gives $(m \times n)$. Most bugs in neural-network code are shape mismatches, so it is a good habit to write the shape next to every tensor.

Two more words. The **transpose** $W^\top$ swaps rows and columns, so an $m \times n$ matrix becomes $n \times m$. In code we often store a batch of vectors as the rows of a matrix $X$ of shape (number of examples) $\times$ (dimension). Then $XW^\top$ transforms all examples in one operation. Computers, and especially GPUs, are extremely fast at this one operation, which is a big reason why neural networks became practical. A **tensor** is the general name for such arrays: a vector is a 1-dimensional tensor, a matrix is 2-dimensional, and a batch of sentences of embeddings is 3-dimensional.

## Probability: describing uncertainty

A language model never knows the next word for sure. It assigns a **probability** to each candidate. A **probability distribution** over a finite set of outcomes is a list of numbers $p_1, \dots, p_K$ with

$$p_k \ge 0 \quad \text{and} \quad \sum_{k=1}^{K} p_k = 1.$$

After "The cat sat on the", a model might give *mat* probability 0.6, *floor* 0.3 and *moon* 0.1.

The **conditional probability** $P(A \mid B)$ is the probability of $A$ when we already know $B$. A language model is precisely a machine for conditional probabilities: $P(\text{next word} \mid \text{previous words})$. As we saw in Chapter 1, the **chain rule of probability** lets us build the probability of a whole sentence from these pieces:

$$P(w_1, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1}).$$

Here $w_t$ is the $t$-th word, $T$ is the length of the sentence and $\prod$ means "multiply all terms".

The **expected value** of a quantity $x$ that takes value $x_k$ with probability $p_k$ is the probability-weighted average $\mathbb{E}[x] = \sum_k p_k x_k$. Training a network means making the *expected* error small over the data, so this notation will appear often.

How does a network produce a distribution? Its last layer outputs arbitrary real numbers called **logits**, which can be negative and need not sum to one. The **softmax** function converts them:

$$\operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

The exponential $e^{z}$ makes every number positive, and dividing by the sum makes them add up to one. A larger logit always gives a larger probability. For logits $(2.0,\ 1.0,\ 0.1)$ the softmax is approximately $(0.659,\ 0.242,\ 0.099)$.

## Logarithms and derivatives

Products of many probabilities become tiny very quickly: fifty factors of 0.1 give $10^{-50}$. So we work with **logarithms**. The logarithm turns products into sums, $\log(ab) = \log a + \log b$, and it is increasing, so a larger probability always has a larger log-probability. In this book, $\log$ means the natural logarithm (base $e$). The log of a probability is always $\le 0$, and $-\log p$ is large when $p$ is small. In Chapter 8 this quantity, $-\log p$, becomes the loss function of language models.

The last tool is the **derivative**. For a function $f(w)$ of one number, the derivative $f'(w)$, also written $\frac{df}{dw}$, tells us how fast $f$ changes when we change $w$ a little:

$$f(w + \Delta) \approx f(w) + f'(w)\,\Delta \quad \text{for small } \Delta.$$

For example, if $f(w) = (w - 3)^2$, then $f'(w) = 2(w - 3)$. At $w = 1$ the derivative is $-4$: increasing $w$ makes $f$ smaller. This is the whole idea of training. We have an error that depends on the weights, and the derivative tells us in which direction to move each weight.

A network has millions of weights, so we need the derivative with respect to each of them. The **partial derivative** $\frac{\partial f}{\partial w_i}$ is the derivative with respect to one variable while the others stay fixed. Collected into one vector, the partial derivatives form the **gradient**:

$$\nabla f = \left( \frac{\partial f}{\partial w_1}, \dots, \frac{\partial f}{\partial w_n} \right).$$

The gradient points in the direction in which $f$ grows fastest. Chapter 8 will walk *against* it to reduce the error, and Chapter 9 will show how to compute it efficiently with the **chain rule** of calculus: if $y = f(g(x))$, then $\frac{dy}{dx} = f'(g(x))\, g'(x)$.

## The same ideas in code

NumPy lets us check all the numbers above in a few lines.

```python
import numpy as np

# Toy 3-dimensional word vectors
cat = np.array([0.9, 0.1, 0.4])
dog = np.array([0.8, 0.2, 0.5])

# Dot product and cosine similarity
dot = cat @ dog
cos = dot / (np.linalg.norm(cat) * np.linalg.norm(dog))
print("dot:", round(dot, 3), "cosine:", round(cos, 3))   # 0.94, 0.985

# A 2x3 matrix maps a 3-dim vector to a 2-dim vector
W = np.array([[1.0, 0.0, -1.0],
              [0.5, 0.5,  0.5]])
print("W @ cat:", W @ cat)                               # [0.5 0.7]

# Softmax turns logits into probabilities
def softmax(z):
    z = z - z.max()          # subtracting the max avoids overflow
    e = np.exp(z)
    return e / e.sum()

p = softmax(np.array([2.0, 1.0, 0.1]))
print("probabilities:", p.round(3), "sum:", p.sum())     # [0.659 0.242 0.099]

# Numerical derivative of f(w) = (w - 3)^2 at w = 1
f = lambda w: (w - 3) ** 2
h = 1e-5
print("derivative:", round((f(1 + h) - f(1 - h)) / (2 * h), 3))  # -4.0
```

Note the line `z - z.max()`. Subtracting the same number from all logits does not change the softmax (the factor $e^{-c}$ cancels in the fraction), but it prevents $e^{z}$ from overflowing for large logits. Real libraries do the same.

## Summary
- A vector is a list of numbers; the dot product $\sum_i a_i b_i$ measures how much two vectors agree, and cosine similarity removes the effect of length.
- A matrix of shape $m \times n$ transforms an $n$-dimensional vector into an $m$-dimensional one; each row acts as a detector.
- A probability distribution is a list of non-negative numbers that sum to one; a language model computes conditional probabilities of the next token.
- Softmax converts arbitrary logits into a probability distribution.
- Logarithms turn products into sums and make tiny probabilities manageable.
- The derivative, and its many-variable form the gradient, tells us how to change weights to reduce an error.

## Exercises
1. Explain in your own words why cosine similarity, and not the plain dot product, is used to compare word embeddings of different lengths.
2. Compute by hand the dot product and cosine similarity of $(1, 2, 0)$ and $(2, 1, 1)$. Then check your answer with NumPy.
3. A matrix $A$ has shape $4 \times 3$ and a matrix $B$ has shape $3 \times 5$. What is the shape of $AB$? Is $BA$ defined? Why or why not?
4. Show that adding the same constant $c$ to all logits does not change the softmax. Then compute the softmax of $(0, 0, \ln 2)$.

## Key terms
| English | Русский | 한국어 |
|---|---|---|
| vector | вектор | 벡터 |
| dimension | размерность | 차원 |
| dot product | скалярное произведение | 내적 |
| norm | норма | 노름 |
| cosine similarity | косинусное сходство | 코사인 유사도 |
| matrix | матрица | 행렬 |
| matrix multiplication | умножение матриц | 행렬 곱셈 |
| transpose | транспонирование | 전치 |
| tensor | тензор | 텐서 |
| probability distribution | распределение вероятностей | 확률 분포 |
| conditional probability | условная вероятность | 조건부 확률 |
| expected value | математическое ожидание | 기댓값 |
| logits | логиты | 로짓 |
| derivative | производная | 도함수(미분) |
| gradient | градиент | 그래디언트(기울기) |

---
[← Previous: Embeddings: Words as Vectors (Word2Vec, GloVe)](05-embeddings.md) · [Contents](README.md) · [Next: Neurons, Layers and Activation Functions →](07-neurons-and-layers.md)
