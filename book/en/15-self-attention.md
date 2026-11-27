# Chapter 15. Self-Attention: Query, Key and Value

> **Part 3 — Sequences and Attention** · Level: Intermediate
>
> **In this chapter you will learn:** how every token of a sequence can attend to every other token in one step · how queries, keys and values are computed and combined in matrix form · why the scores are divided by $\sqrt{d_k}$, and what self-attention gains and loses compared with an RNN

## Which word does "it" refer to?

Read this sentence: "The animal didn't cross the street because **it** was too tired." What does "it" refer to? Almost every reader answers "the animal" immediately. Now change one word: "The animal didn't cross the street because **it** was too wide." Now "it" is the street.

To build a good representation of the word "it", a model must look at other words in the *same* sentence — "animal", "street", "tired", "wide" — and decide which of them matter. An RNN (Chapter 11) can only do this indirectly: information about "animal" has to survive in the hidden state through every step until "it" arrives. In Chapter 14 we saw a much more direct tool: attention, which lets one vector look at a whole set of vectors and take what it needs. There the decoder attended to the encoder. What if every word of a sentence attends to all the words of the same sentence?

This is **self-attention**. For each position, it computes a weighted mixture of the representations of all positions, with weights that depend on how relevant the other words are. After one self-attention layer, the vector for "it" can already contain information from "animal" — not after a long chain of recurrent steps, but in a single operation. Self-attention is the central component of the Transformer, introduced by Vaswani et al. (2017) in the paper "Attention Is All You Need".

## Three roles: query, key and value

In Chapter 14 the encoder states played two roles at once: they were compared with the decoder state (as keys) and they were mixed into the context (as values). In self-attention we separate the roles cleanly and give each token **three** different vectors.

An analogy helps. Think of a library search. You come with a **query** — what you are looking for. Each book has a **key** — the label on the spine that describes it. You compare your query with every key; the better the match, the more attention you give to that book. What you finally take home is the book's **value** — its contents. The key is what makes an item *findable*; the value is what it *delivers*. They need not be the same.

In self-attention each token plays all three roles. Let $x_i \in \mathbb{R}^{d_{model}}$ be the representation (for example, the embedding) of token $i$. We compute

$$q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V$$

where $W_Q$ and $W_K$ are learned matrices of size $d_{model} \times d_k$, and $W_V$ has size $d_{model} \times d_v$. (We write vectors as rows here, as is usual in Transformer code.) The query $q_i$ says "what token $i$ is looking for", the key $k_j$ says "what token $j$ offers for matching", and the value $v_j$ is "the information token $j$ passes on".

The output for token $i$ follows the same three steps as in Chapter 14:

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}, \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{l=1}^{n} \exp(e_{il})}, \qquad z_i = \sum_{j=1}^{n} \alpha_{ij}\, v_j$$

Here $n$ is the number of tokens, $e_{ij}$ is the score of token $j$ from the point of view of token $i$, the weights $\alpha_{ij}$ in each row sum to 1, and $z_i$ is the new, context-aware representation of token $i$. Notice that token $i$ also attends to itself ($j = i$). The division by $\sqrt{d_k}$ is explained below.

Because $W_Q$, $W_K$ and $W_V$ are learned, the model decides what "looking for" and "offering" mean. In a trained model, one might find that the query of a pronoun matches well with the keys of nouns, so that "it" collects information from "animal". Nobody programs this; it emerges from training on next-token prediction.

## A worked example with three tokens

Let us compute one output by hand. Take three tokens and $d_k = d_v = 2$. Suppose the projections have already given us the query of token 1 and the keys and values of all three tokens:

$$q_1 = (1, 0)$$

$$k_1 = (1, 0), \quad k_2 = (0, 1), \quad k_3 = (1, 1)$$

$$v_1 = (1, 0), \quad v_2 = (0, 1), \quad v_3 = (2, 2)$$

**Scores.** The dot products are $q_1 \cdot k_1 = 1$, $q_1 \cdot k_2 = 0$, $q_1 \cdot k_3 = 1$. Dividing by $\sqrt{2} \approx 1.414$ gives $e_1 = (0.707, \; 0, \; 0.707)$.

**Weights.** $e^{0.707} \approx 2.028$ and $e^0 = 1$, so the sum is $5.056$ and

$$\alpha_1 = (0.401, \; 0.198, \; 0.401)$$

**Output.**

$$z_1 = 0.401 \cdot (1, 0) + 0.198 \cdot (0, 1) + 0.401 \cdot (2, 2) = (1.203, \; 1.000)$$

Token 1 takes most of its new representation from tokens 1 and 3, whose keys match its query, and a little from token 2. The same computation is done for every token, each with its own query.

## Self-attention in matrix form

Doing this token by token would be slow. The beauty of self-attention is that it can be written with a few matrix multiplications. Stack the token vectors as rows of a matrix $X \in \mathbb{R}^{n \times d_{model}}$. Then

$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V$$

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Here $Q K^\top$ is an $n \times n$ matrix that contains all the scores $q_i \cdot k_j$ at once: row $i$ describes how token $i$ sees all the others. The softmax is applied to each row separately, giving the **attention matrix** $A$, whose rows sum to 1. Multiplying $A$ by $V$ ($n \times d_v$) mixes the values and gives the output $Z \in \mathbb{R}^{n \times d_v}$, one row per token. This formula is called **scaled dot-product attention**.

The code below implements it in NumPy:

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def self_attention(X, W_q, W_k, W_v):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v       # (n, d_k), (n, d_k), (n, d_v)
    scores = Q @ K.T / np.sqrt(K.shape[-1])   # (n, n): every token vs every token
    A = softmax(scores)                       # each row sums to 1
    return A @ V, A

rng = np.random.default_rng(0)
n, d_model, d_k = 3, 4, 2                     # 3 tokens, model size 4, head size 2
X = rng.normal(size=(n, d_model))             # token embeddings
W_q, W_k, W_v = (rng.normal(size=(d_model, d_k)) for _ in range(3))

out, A = self_attention(X, W_q, W_k, W_v)
print("attention matrix:\n", np.round(A, 3))
print("output shape:", out.shape)

# Shuffle the tokens: the outputs are shuffled in the same way
perm = [2, 0, 1]
out_p, _ = self_attention(X[perm], W_q, W_k, W_v)
print("permutation equivariant:", np.allclose(out_p, out[perm]))
```

There are no loops over positions: all tokens are processed at once. This is exactly the kind of computation GPUs are built for. (The last lines of the code illustrate an important property that we discuss in the final section.)

## Why divide by the square root of $d_k$?

In the exercises of Chapter 14 we saw that the *scale* of the scores controls how sharp the softmax is: multiply all scores by 10 and almost all the weight goes to the largest one; multiply them by 0.1 and the weights become nearly uniform.

Now consider the dot product $q \cdot k = \sum_{m=1}^{d_k} q_m k_m$. Suppose the components $q_m$ and $k_m$ are independent random numbers with mean 0 and variance 1, which is roughly the situation at initialization. Each product $q_m k_m$ then has mean 0 and variance 1, and the sum of $d_k$ such terms has variance $d_k$. So the typical size (standard deviation) of a score grows like $\sqrt{d_k}$. With $d_k = 64$, a common head size, scores of $\pm 8$ or more are normal.

Scores that large push the softmax into saturation: one weight is almost 1 and the others are almost 0. In this region the gradient of the softmax is tiny, and learning becomes slow — a problem similar to the vanishing gradients of Chapter 11. Dividing by $\sqrt{d_k}$ brings the variance of the scores back to about 1, whatever the dimension. This simple rescaling is the "scaled" in scaled dot-product attention. Chapter 16 builds on it with masking and multiple attention heads.

## What self-attention gains and what it loses

Compare self-attention with an RNN on a sentence of $n$ tokens.

**Parallelism.** An RNN must compute $h_1, h_2, \dots, h_n$ one after another. Self-attention computes all outputs at once with matrix products. This is the main reason the Transformer could be trained on far more data than recurrent models.

**Path length.** In an RNN, information from token 1 reaches token $n$ only after passing through $n - 1$ recurrent steps, being transformed (and possibly weakened) at each one. In self-attention, every token is connected to every other token directly, so the maximum path length is 1. Long-range dependencies, like "animal" and "it", become easy to learn.

**Cost.** The attention matrix has $n \times n$ entries. Computation and memory therefore grow as $O(n^2)$ with the sequence length: doubling the length makes the attention matrix four times larger. For sentences this is fine; for very long contexts it becomes the main bottleneck, and many methods in Parts 4 and 8 aim to reduce it. An RNN, by contrast, costs $O(n)$ steps.

**Order.** This is the most surprising point. Look at the formula again: nothing in it refers to positions. If we shuffle the input tokens, every query, key and value is shuffled the same way, the attention matrix is shuffled in rows and columns, and the outputs come out in shuffled order but are otherwise *identical*. The code above checks this: `permutation equivariant: True`. This property is called **permutation equivariance**. For self-attention, "The dog bit the man" and "The man bit the dog" are the same set of words — exactly the problem we started Chapter 11 with! An RNN knows the order because it reads left to right; self-attention does not. The solution is to add information about each token's position to its representation before attention. That is the topic of **positional encoding** in Chapter 17.

Finally, in a language model that predicts the next token, a token must not attend to tokens that come *after* it — that would be cheating. This is handled by a **causal mask** that sets the scores of future positions to $-\infty$ before the softmax, as we will see in Chapter 16. With these two additions — positions and masking — and with several attention heads working in parallel, self-attention becomes the engine of the Transformer, which is the subject of Part 4.

## Summary

- Self-attention lets every token of a sequence attend to every token of the same sequence, including itself, in a single step.
- Each token gets a query $q_i = x_i W_Q$, a key $k_i = x_i W_K$ and a value $v_i = x_i W_V$; queries are matched with keys, and values are mixed.
- In matrix form: $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}(Q K^\top / \sqrt{d_k})\, V$, with an $n \times n$ attention matrix whose rows sum to 1.
- Dividing by $\sqrt{d_k}$ keeps the variance of the scores near 1 and prevents the softmax from saturating.
- Compared with RNNs, self-attention is parallel and has path length 1, but costs $O(n^2)$ in sequence length.
- Self-attention is permutation equivariant: it ignores word order, so positional information must be added (Chapter 17).

## Exercises

1. Explain in your own words the difference between a query, a key and a value, using the library analogy or your own example. Why is it useful to have separate key and value vectors instead of one?
2. Repeat the worked example for token 3 with the query $q_3 = (1, 1)$ and the same keys and values. Compute the scaled scores, the weights and the output $z_3$.
3. A sequence has 1,000 tokens. How many entries does the attention matrix have? How many if the sequence has 4,000 tokens? By what factor did the length grow, and by what factor did the matrix grow?
4. Using the code above, generate random vectors $q, k \in \mathbb{R}^{d}$ with independent standard normal components for $d = 4, 64, 512$ (say 10,000 pairs each) and print the standard deviation of $q \cdot k$ and of $q \cdot k / \sqrt{d}$. Do the results match the argument in this chapter?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| self-attention | самовнимание | 셀프 어텐션 |
| query | запрос | 쿼리 |
| key | ключ | 키 |
| value | значение | 값 |
| projection matrix | матрица проекции | 투영 행렬 |
| attention matrix | матрица внимания | 어텐션 행렬 |
| scaled dot-product attention | внимание на основе масштабированного скалярного произведения | 스케일드 닷 프로덕트 어텐션 |
| scaling factor | масштабирующий коэффициент | 스케일링 계수 |
| saturation | насыщение | 포화 |
| path length | длина пути | 경로 길이 |
| quadratic complexity | квадратичная сложность | 이차 복잡도 |
| permutation equivariance | эквивариантность к перестановкам | 순열 등변성 |
| positional encoding | позиционное кодирование | 위치 인코딩 |
| causal mask | каузальная маска | 인과 마스크 |
| transformer | трансформер | 트랜스포머 |

---
[← Previous: The Idea of Attention (Bahdanau, Luong)](14-the-idea-of-attention.md) · [Contents](README.md) · [Next: Scaled Dot-Product and Multi-Head Attention, Masking →](16-multi-head-attention.md)
