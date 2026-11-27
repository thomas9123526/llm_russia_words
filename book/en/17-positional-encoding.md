# Chapter 17. Positional Information: Sinusoidal, Learned, RoPE, ALiBi

> **Part 4 — The Transformer** · Level: Intermediate
>
> **In this chapter you will learn:** why self-attention needs to be told the order of tokens · how absolute positions are added with sinusoidal or learned vectors · how RoPE and ALiBi encode relative distance directly inside attention

## Dog bites man, man bites dog

"The dog bit the man" and "the man bit the dog" contain exactly the same tokens. A newspaper would print only one of them. For a human the difference is obvious; for self-attention, as we saw at the end of Chapter 16, it is invisible.

Let us make this precise. In self-attention the output for token $i$ is $\sum_j \alpha_{ij} v_j$, where the weights $\alpha_{ij}$ depend only on the dot products $q_i \cdot k_j$. Nothing in this formula mentions *where* token $j$ is. If we swap "dog" and "man", the vector computed for "bit" is exactly the same, because it is a weighted sum over the same set of keys and values. In other words, the representation of a token does not depend on the order of the other tokens; this property is called **permutation invariance**. An RNN (Chapter 11) did not have this problem: it read tokens one by one, so order was built into the computation. The Transformer gave up this sequential reading to gain parallelism, and must now get order back in another way.

The solution is to inject **positional information**. There are two big families. *Absolute* methods give every position $0, 1, 2, \dots$ its own vector and add it to the token embedding. *Relative* methods change the attention computation itself so that the score between two tokens depends on the distance between them. We will look at two absolute methods (sinusoidal and learned) and two relative ones (RoPE and ALiBi).

## Absolute positions: sinusoidal encoding

The original Transformer used a fixed **sinusoidal positional encoding**. For position $pos$ and dimension index $i$ it defines a vector $PE_{pos}$ of width $d$ (the model dimension):

$$
PE_{pos,\,2i} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{pos,\,2i+1} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Here $i$ runs from $0$ to $d/2 - 1$, so every pair of dimensions $(2i, 2i+1)$ is a sine and a cosine of the same **frequency** $\omega_i = 1/10000^{2i/d}$. The vector $PE_{pos}$ is simply added to the embedding of the token at position $pos$ before the first layer.

The intuition is a clock with many hands. The first pair of dimensions turns fast: its angle grows by one radian per position, a full turn about every 6 tokens. The last pair turns very slowly: its **wavelength** is about $2\pi \cdot 10000$ positions. Together, the hands give every position a unique "time" reading. Nearby positions have similar vectors in the slow dimensions and differ in the fast ones, just as 3:15 and 3:16 differ only in the minute hand.

The authors chose sines and cosines for a further reason. For a fixed offset $k$, the pair $(\sin \omega(pos+k), \cos \omega(pos+k))$ is obtained from $(\sin \omega\, pos, \cos \omega\, pos)$ by a rotation by the angle $\omega k$. So "$k$ positions later" is a linear transformation that does not depend on $pos$, which the model can in principle learn to use. Keep this rotation in mind: it will return in RoPE.

## Absolute positions: learned embeddings

The simplest alternative is to let the model learn the position vectors. We create a matrix with one row per position, exactly like the token embedding matrix of Chapter 5, and train it together with the rest of the network. In PyTorch this is one line: `nn.Embedding(max_len, d)`. BERT and GPT-2 use **learned positional embeddings**, and in the original Transformer paper they performed about as well as the sinusoidal ones.

Learned positions are flexible, but they have a clear limitation. The table has exactly `max_len` rows. Position `max_len + 1` has no vector at all, so the model cannot be used on longer texts, and positions near the end of the table are seen less often during training. The sinusoidal formula can be evaluated for any position, but in practice models trained with it also degrade on sequences much longer than those seen in training. The ability to work on longer sequences than in training is called **length extrapolation**, and it is a major motivation for the relative methods.

There is also a conceptual argument. In "the dog bit the man", what matters for "bit" is that "dog" is *one token to the left*, not that "dog" is at position 1. If the same sentence appears at position 1001 of a document, its meaning does not change. **Relative positions** describe exactly this.

## RoPE: rotate queries and keys

**Rotary position embedding (RoPE)**, introduced by Su et al. in the RoFormer paper, is today the most common choice in open decoder models such as LLaMA (Chapter 25). Its idea is elegant: instead of *adding* a position vector to the input, we *rotate* each query and key by an angle proportional to its position.

Consider first a 2-dimensional query $q$ at position $m$ and a key $k$ at position $n$. Let $R(\phi)$ be the $2 \times 2$ **rotation matrix** by the angle $\phi$. RoPE replaces $q$ by $R(m\theta)\,q$ and $k$ by $R(n\theta)\,k$, where $\theta$ is a fixed frequency. The attention score becomes

$$
\big(R(m\theta)\,q\big)^\top \big(R(n\theta)\,k\big) = q^\top R\big((n-m)\theta\big)\, k .
$$

This follows from two facts about rotations: $R(\phi)^\top = R(-\phi)$, and rotations add up, $R(-m\theta)R(n\theta) = R((n-m)\theta)$. The result depends on the positions only through the difference $n - m$. The absolute positions have disappeared; only the relative distance remains.

A tiny numeric example. Let $q = k = (1, 0)$ and $\theta = 30°$. At positions $m = 1, n = 0$ the query is rotated by $30°$ and the key by $0°$; their dot product is $\cos 30° = 0.866$. At positions $m = 5, n = 4$ they are rotated by $150°$ and $120°$, the angle between them is again $30°$, and the dot product is again $0.866$.

For a real query of width $d$, RoPE splits it into $d/2$ pairs and rotates pair $i$ by the angle $m\theta_i$, with the same frequencies $\theta_i = 1/10000^{2i/d}$ as the sinusoidal encoding. Fast pairs react to small distances, slow pairs to large ones. The values $v$ are not rotated, and nothing is added to the embeddings. The code below implements both the sinusoidal encoding and RoPE and checks the relative property:

```python
import torch

def sinusoidal(T, d):
    pos = torch.arange(T).unsqueeze(1)              # positions 0..T-1, shape (T, 1)
    i = torch.arange(0, d, 2)                       # even dimensions 0, 2, 4, ...
    freq = 1.0 / (10000 ** (i / d))                 # one frequency per pair of dimensions
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe

def rope(x, pos):
    # rotate each pair (x0, x1), (x2, x3), ... by the angle pos * theta_i
    d = x.size(-1)
    theta = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))
    cos, sin = torch.cos(pos * theta), torch.sin(pos * theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

print(sinusoidal(4, 6).round(decimals=3))
torch.manual_seed(0)
q, k = torch.randn(8), torch.randn(8)
# the score depends only on the distance m - n = 3, not on m and n themselves
for m, n in [(5, 2), (10, 7), (103, 100)]:
    print(m, n, round(torch.dot(rope(q, m), rope(k, n)).item(), 3))
```

In the printed $4 \times 6$ table, row 0 is `[0, 1, 0, 1, 0, 1]` (sine of 0 and cosine of 0), and row 1 begins with `0.841, 0.540` — that is $\sin 1$ and $\cos 1$ — while its last pair is almost unchanged (`0.002, 1.0`): the slow hand has barely moved. The three RoPE scores are all equal (`1.178` for this seed), although the absolute positions differ by a factor of twenty.

## ALiBi: a penalty for distance

**ALiBi** (Attention with Linear Biases, Press et al., "Train Short, Test Long") is even simpler. It uses no position vectors at all. Instead, it adds a fixed penalty to each attention score, proportional to the distance between the tokens:

$$
\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - s \cdot (i - j), \qquad j \le i
$$

Here $i$ is the position of the query, $j \le i$ the position of the key (ALiBi was designed for causal models), and $s > 0$ is a **slope** that differs from head to head. For 8 heads the slopes form a geometric sequence $\tfrac12, \tfrac14, \dots, \tfrac{1}{256}$. A head with a large slope strongly prefers nearby tokens and acts locally; a head with a small slope can still look far back. The slopes are not learned.

The bias encodes a simple prior: closer tokens usually matter more. Because the penalty is defined for any distance, a model trained on short sequences can be run on longer ones, and the authors showed that ALiBi extrapolates to longer inputs much better than sinusoidal encoding. RoPE does not extrapolate well by itself, so models that use it are often extended with **position interpolation** and related tricks, which squeeze longer sequences into the range of rotation angles seen during training (we return to long contexts in Chapter 25).

## Choosing a method

To summarize the landscape: sinusoidal and learned encodings are *absolute* and are added once at the input; RoPE and ALiBi are *relative* and act inside every attention layer, on queries and keys or on scores. Most modern decoder-only LLMs use RoPE; some use ALiBi; BERT-style encoders often keep learned absolute positions. None of these choices changes the overall architecture: the input is still a sequence of vectors, and the next chapter assembles attention and other parts into a complete Transformer block.

## Summary

- Without positional information, self-attention treats a sentence as an unordered set of tokens.
- Sinusoidal encoding adds fixed sine and cosine vectors of many frequencies to the embeddings; a shift by $k$ positions is a rotation.
- Learned positional embeddings are a trainable table of vectors; they are simple but limited to the maximum length seen in training.
- RoPE rotates queries and keys by position-dependent angles, so the attention score depends only on relative distance.
- ALiBi subtracts a head-specific linear penalty for distance from the scores and extrapolates well to longer sequences.

## Exercises

1. Explain in your own words why self-attention without positional information gives the same vector for "bit" in "the dog bit the man" and "the man bit the dog".
2. For $d = 512$, compute the frequency $\omega_i$ and the wavelength $2\pi/\omega_i$ of the first pair ($i = 0$) and the last pair ($i = 255$) of the sinusoidal encoding.
3. With $q = k = (1, 0)$ and $\theta = 45°$, compute the RoPE score for $(m, n) = (2, 0)$ and $(m, n) = (7, 5)$. Then compute it for $(m, n) = (4, 0)$.
4. Write a function that builds the ALiBi bias matrix of shape (heads, $T$, $T$) for $T = 5$ and 4 heads with slopes $\tfrac14, \tfrac1{16}, \tfrac1{64}, \tfrac1{256}$, and add $-\infty$ above the diagonal.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| positional encoding | позиционное кодирование | 위치 인코딩 |
| permutation invariance | инвариантность к перестановкам | 순열 불변성 |
| absolute position | абсолютная позиция | 절대 위치 |
| relative position | относительная позиция | 상대 위치 |
| sinusoidal positional encoding | синусоидальное позиционное кодирование | 사인파 위치 인코딩 |
| frequency | частота | 주파수 |
| wavelength | длина волны | 파장 |
| learned positional embeddings | обучаемые позиционные эмбеддинги | 학습형 위치 임베딩 |
| length extrapolation | экстраполяция по длине | 길이 외삽 |
| rotary position embedding (RoPE) | ротационное позиционное кодирование (RoPE) | 회전 위치 임베딩(RoPE) |
| rotation matrix | матрица поворота | 회전 행렬 |
| ALiBi (attention with linear biases) | ALiBi (внимание с линейными смещениями) | ALiBi(선형 편향 어텐션) |
| slope | наклон | 기울기 |
| position interpolation | интерполяция позиций | 위치 보간 |

---
[← Previous: Scaled Dot-Product and Multi-Head Attention, Masking](16-multi-head-attention.md) · [Contents](README.md) · [Next: The Transformer Block: Residuals, LayerNorm/RMSNorm, FFN, GLU →](18-transformer-block.md)
