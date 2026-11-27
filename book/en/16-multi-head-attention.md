# Chapter 16. Scaled Dot-Product and Multi-Head Attention, Masking

> **Part 4 — The Transformer** · Level: Intermediate
>
> **In this chapter you will learn:** why attention scores are divided by $\sqrt{d_k}$ · how several attention heads look at a sentence in parallel · how masks stop a token from seeing the future or the padding

## From one question to many questions

Take the sentence *"The cat that the dog chased ran away."* When the model processes the word *ran*, it needs at least two different pieces of information. Grammatically, it must find the subject: who ran? The answer is *cat*, five words back. Semantically, it may also want to know what happened just before: *chased*, *dog*. These are two different questions about the same sentence.

In Chapter 15 we built self-attention: every token produces a query $q$, a key $k$ and a value $v$; the query is compared with all keys, the scores become weights through softmax, and the output is a weighted average of the values. One such attention operation gives each token **one** weighted average, so it can answer roughly one "question" at a time. If the subject and the recent action are both important, a single set of weights has to compromise between them.

The Transformer of Vaswani et al. ("Attention Is All You Need", 2017) solves this in a simple way: it runs several attention operations in parallel, each with its own projections, and then combines the results. This is **multi-head attention**. Before we build it, we need to fix one numerical detail of single-head attention — the scaling — and one practical detail — the mask. Together these three ideas are the core of every Transformer in this book.

## Why we divide by $\sqrt{d_k}$

Recall the attention formula from Chapter 15, now with its full name, **scaled dot-product attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Here $Q$ is the matrix of queries (one row per token), $K$ the matrix of keys, $V$ the matrix of values, and $d_k$ the length of each query and key vector. The product $QK^\top$ contains all pairwise dot products $q_i \cdot k_j$: row $i$ tells how well token $i$'s query matches every key.

Why the division by $\sqrt{d_k}$? Intuition first. A dot product is a sum of $d_k$ terms $q_1k_1 + q_2k_2 + \dots$. The more terms we add, the larger the typical size of the sum. Suppose each component of $q$ and $k$ is a random number with mean 0 and variance 1, independent of the others. Each product $q_m k_m$ then has mean 0 and variance 1, and the variance of a sum of independent terms is the sum of variances:

$$
\operatorname{Var}(q \cdot k) = \sum_{m=1}^{d_k} \operatorname{Var}(q_m k_m) = d_k .
$$

So the standard deviation of the score grows like $\sqrt{d_k}$. With $d_k = 64$, typical scores are around $\pm 8$; with $d_k = 512$, around $\pm 22$. This is a problem for softmax. Softmax of large numbers is almost a one-hot vector: $\text{softmax}(8, 0, 0) \approx (0.9993, 0.0003, 0.0003)$. Such a distribution is **saturated**: one weight is almost 1, the rest almost 0, and the gradients through the small weights are close to zero (recall the vanishing gradients of Chapter 11). Training becomes slow and unstable.

Dividing by $\sqrt{d_k}$ brings the standard deviation back to about 1, whatever the dimension. We can check this numerically:

```python
import torch

torch.manual_seed(0)
for d in (4, 64, 512):
    q, k = torch.randn(10000, d), torch.randn(10000, d)  # random unit-variance vectors
    s = (q * k).sum(-1)                                  # 10,000 dot products
    print(d, round(s.std().item(), 2), round((s / d ** 0.5).std().item(), 2))
```

The output is approximately `4 1.99 1.0`, `64 8.01 1.0`, `512 22.49 0.99`: the raw standard deviation follows $\sqrt{d}$ (2, 8, 22.6), while the scaled one stays at 1. That is the whole reason for the word "scaled".

## Multi-head attention

Now the main idea. Instead of one attention operation on vectors of width $d_{\text{model}}$ (the **model dimension**, for example 512), we run $h$ smaller ones, called **heads**, each on vectors of width $d_k = d_{\text{model}} / h$ (for example $512 / 8 = 64$). Head $i$ has its own projection matrices $W_i^Q, W_i^K, W_i^V$ of size $d_{\text{model}} \times d_k$:

$$
\text{head}_i = \text{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V)
$$

Here $X$ is the matrix of input token vectors (one row per token). Each head projects the same input into its own small **subspace** and computes its own attention weights. One head may learn to connect verbs with their subjects, another to look at the previous token, a third to follow quotation marks. Nobody tells the heads what to do; the specialization emerges from training, and researchers who visualize attention maps (Chapter 40) find both interpretable and uninterpretable heads.

The outputs of all heads are then **concatenated** back into one vector of width $h \cdot d_k = d_{\text{model}}$ and mixed by an output matrix $W^O$ of size $d_{\text{model}} \times d_{\text{model}}$:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O
$$

Two remarks help to see why this design is attractive.

First, the cost. Because each head works in a space $h$ times smaller, the total number of parameters and operations is about the same as for one big head: $h$ matrices of size $d_{\text{model}} \times d_k$ together form one matrix of size $d_{\text{model}} \times d_{\text{model}}$. We get several "questions" almost for free.

Second, the implementation. In code we do not create $h$ separate layers. We apply one large linear layer that produces all queries, keys and values at once, and then **reshape** the result: a tensor of shape (batch, tokens, $d_{\text{model}}$) becomes (batch, heads, tokens, $d_k$). After attention we reverse the reshape — this *is* the concatenation — and apply $W^O$.

## Masking: what a token is allowed to see

Attention lets every token look at every other token. Sometimes this is wrong, and we need a **mask** that forbids some connections. There are two common cases.

**The causal mask.** A language model that generates text from left to right (Chapter 1, and in detail Chapter 23) must predict token $t+1$ using only tokens $1, \dots, t$. During training we feed the whole sentence at once for efficiency, so without a mask token 3 could simply "look" at token 4 and copy the answer. The **causal mask** forbids attention to later positions. It is a lower-triangular matrix: position $i$ may attend to position $j$ only if $j \le i$.

How do we forbid a connection? We cannot just set a weight to zero after softmax, because then the row no longer sums to 1. Instead we set the *score* to $-\infty$ before softmax. Since $e^{-\infty} = 0$, the forbidden position gets weight exactly 0, and the remaining weights are renormalized automatically.

A small worked example with 3 tokens. Suppose the scaled scores of the third token are $(2, 1, 0)$. Softmax gives $e^2 : e^1 : e^0 = 7.389 : 2.718 : 1$, that is $(0.665, 0.245, 0.090)$. Now take the second token with scores $(2, 1, 0)$ as well. Position 3 is in its future, so the mask turns the scores into $(2, 1, -\infty)$, and softmax gives $7.389 : 2.718 : 0$, that is $(0.731, 0.269, 0)$. The first token can see only itself, so its weights are $(1, 0, 0)$.

**The padding mask.** Sentences in a batch have different lengths, so short ones are filled with a special padding token (Chapter 10). These positions carry no meaning, and real tokens should not attend to them. A **padding mask** sets their scores to $-\infty$ in the same way. In practice the causal mask and the padding mask are often combined with a logical AND. One caution: if a row is masked *completely*, softmax divides 0 by 0 and returns NaN; libraries handle this case specially, and your own code should avoid it.

Here is a compact implementation of everything in this chapter:

```python
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None):
    # q, k, v: (batch, heads, seq, d_k); mask: True = "may attend"
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5  # (batch, heads, seq, seq)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)               # each row sums to 1
    return weights @ v, weights
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)  # W_Q, W_K, W_V in one matrix
        self.out = torch.nn.Linear(d_model, d_model)      # W_O
    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, h, T, d_k): split the width into h heads
        q, k, v = [t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v)]
        y, w = attention(q, k, v, mask)
        y = y.transpose(1, 2).reshape(B, T, D)            # concatenate the heads
        return self.out(y), w

torch.manual_seed(0)
x = torch.randn(1, 4, 8)                                  # 1 sentence, 4 tokens, d_model = 8
causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))   # lower-triangular mask
y, w = MultiHeadAttention(8, 2)(x, causal)
print(y.shape, w.shape, w[0, 0].detach().round(decimals=2), sep="\n")  # head 0: zeros above the diagonal
```

The output shapes are `torch.Size([1, 4, 8])` and `torch.Size([1, 2, 4, 4])`: the output has the same shape as the input, and there is one $4 \times 4$ weight matrix per head. The printed matrix of head 0 has zeros above the diagonal, and each row sums to 1. Its first row is `[1, 0, 0, 0]`: the first token can only attend to itself.

PyTorch also has a built-in function, `torch.nn.functional.scaled_dot_product_attention`, which computes the same thing much faster; we will use it in Chapter 20. Writing it by hand once is the best way to understand it.

## The cost of attention

One more property deserves attention. The score matrix $QK^\top$ has size $n \times n$ for a sequence of $n$ tokens, in every head and every layer. Doubling the context length therefore makes this part four times more expensive in both time and memory. This **quadratic complexity** is the main reason why long contexts are hard, and it motivates many ideas later in the book: the KV cache and efficient attention kernels (Chapter 36), and grouped-query attention (Chapter 25), which shares keys and values between heads.

Notice also what attention does *not* know: the order of tokens. If we shuffle the input rows, the outputs are shuffled in the same way, but otherwise unchanged. For attention, "the dog chased the cat" and "the cat chased the dog" contain the same set of tokens. The causal mask adds some order information, but not enough. How to tell the model where each token is located is the subject of the next chapter.

## Summary

- Scaled dot-product attention divides scores by $\sqrt{d_k}$, because the variance of a dot product of random vectors grows linearly with $d_k$ and large scores saturate softmax.
- Multi-head attention runs $h$ attention heads in parallel, each in its own subspace of width $d_{\text{model}}/h$, then concatenates them and mixes them with $W^O$.
- Several heads cost about the same as one wide head, but let the model attend to different relations at once.
- A mask sets forbidden scores to $-\infty$ before softmax: the causal mask hides future tokens, the padding mask hides padding.
- Attention has quadratic cost in sequence length and by itself ignores token order.

## Exercises

1. Explain in your own words why we set masked scores to $-\infty$ *before* softmax rather than setting weights to 0 *after* softmax.
2. Compute the softmax of $(4, 0)$ and of $(4/\sqrt{16}, 0) = (1, 0)$. How does scaling change the distribution?
3. A model has $d_{\text{model}} = 768$ and 12 heads. What is $d_k$? How many parameters do $W^Q, W^K, W^V, W^O$ have together (without biases)?
4. Modify the code to build a padding mask for a batch of two sentences of lengths 4 and 2 (padded to 4), combine it with the causal mask, and print the weights of the second sentence.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| scaled dot-product attention | внимание на основе масштабированного скалярного произведения | 스케일드 닷-프로덕트 어텐션 |
| dot product | скалярное произведение | 내적 |
| multi-head attention | многоголовое внимание | 멀티헤드 어텐션 |
| attention head | голова внимания | 어텐션 헤드 |
| model dimension | размерность модели | 모델 차원 |
| variance | дисперсия | 분산 |
| saturation | насыщение | 포화 |
| subspace | подпространство | 부분 공간 |
| concatenation | конкатенация | 연결(concatenation) |
| projection matrix | матрица проекции | 투영 행렬 |
| attention mask | маска внимания | 어텐션 마스크 |
| causal mask | каузальная маска | 인과 마스크 |
| padding mask | маска паддинга | 패딩 마스크 |
| quadratic complexity | квадратичная сложность | 이차 복잡도 |

---
[← Previous: Self-Attention: Query, Key and Value](15-self-attention.md) · [Contents](README.md) · [Next: Positional Information: Sinusoidal, Learned, RoPE, ALiBi →](17-positional-encoding.md)
