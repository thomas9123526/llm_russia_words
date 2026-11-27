# Chapter 18. The Transformer Block: Residuals, LayerNorm/RMSNorm, FFN, GLU

> **Part 4 — The Transformer** · Level: Intermediate
>
> **In this chapter you will learn:** which parts make up one Transformer block and why each is needed · how residual connections and normalization make deep stacks trainable · what the feed-forward network does and why modern models use gated variants such as SwiGLU

## Why attention alone is not enough

Suppose we take the multi-head attention layer of Chapter 16, add the positional information of Chapter 17, and simply stack 24 such layers. Will we get a good language model? No, for two reasons.

The first reason is about *what* the layers compute. Attention outputs a weighted average of value vectors. Averaging is good for moving information *between* tokens — "copy something from the subject into the verb" — but it does little to *transform* the information inside each token. Apart from the softmax, attention is mostly linear in the values. We also need a component that takes each token's vector and processes it with a real nonlinearity, as the multilayer networks of Chapter 7 did.

The second reason is about *training*. As we saw in Chapters 9 and 11, very deep networks suffer from vanishing and exploding gradients and from activations whose scale drifts from layer to layer. A stack of 24 layers without special care would barely train.

The **Transformer block** solves both problems. It has two **sublayers**: multi-head attention, which lets tokens *communicate*, and a **feed-forward network (FFN)**, which lets each token *compute*. Around each sublayer it puts two pieces of engineering: a **residual connection** and a **normalization** layer. A model is just the same block repeated $N$ times. In this chapter we take the block apart.

## Residual connections and the residual stream

A residual connection, introduced for image networks by He et al. (ResNet), is a simple idea: instead of replacing the input $x$ by a sublayer's output $F(x)$, we *add* the output to the input:

$$
x_{\text{new}} = x + F(x)
$$

Why does this help? Look at the gradient. By the chain rule (Chapter 9), $\partial x_{\text{new}} / \partial x = I + \partial F / \partial x$, where $I$ is the identity matrix. Even if the sublayer's own derivative is tiny, the identity term passes the gradient back unchanged. With residuals in every block, there is a direct "highway" from the loss to the first layer, and gradients cannot vanish just because the network is deep. A second benefit: at the start of training, when $F$ is close to zero, each block is close to the identity function, so a deep network starts as a nearly trivial one and gradually learns to use its layers.

Residuals also give a useful way to think about the whole model. Imagine one vector per token flowing from the embedding layer to the output. Every attention and FFN sublayer *reads* from this vector and *writes* an update back by addition. This running sum is called the **residual stream**. In the residual-stream picture, layers do not pass messages to the next layer only; they write into a shared memory that all later layers can read. This picture is very common in interpretability research (Chapter 40).

## LayerNorm and RMSNorm

Adding updates layer after layer can make the vectors in the residual stream grow or change scale. Neural networks train best when the inputs of each layer have a stable scale, so the Transformer normalizes vectors before (or after) each sublayer.

**Layer normalization (LayerNorm)**, proposed by Ba, Kiros and Hinton, normalizes each token's vector separately over its $d$ features:

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad \mu = \frac1d \sum_{m=1}^{d} x_m,\quad \sigma^2 = \frac1d \sum_{m=1}^{d} (x_m - \mu)^2
$$

Here $\mu$ and $\sigma^2$ are the mean and variance of the components of $x$, $\epsilon$ is a tiny number that prevents division by zero, $\gamma$ and $\beta$ are learned vectors of size $d$ (a scale and a shift), and $\odot$ means **element-wise multiplication**. Unlike batch normalization (Chapter 10), LayerNorm does not depend on the other examples in the batch, which makes it well suited to sequences of different lengths and to generation one token at a time.

**RMSNorm** (Zhang and Sennrich) simplifies LayerNorm by dropping the mean subtraction and the shift $\beta$. It only divides by the **root mean square** of the components:

$$
\text{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\frac1d \sum_{m=1}^{d} x_m^2 + \epsilon}}
$$

where $g$ is a learned gain vector. RMSNorm is slightly cheaper and works as well in practice, so it is used in LLaMA and many other recent models.

A worked example with $x = (1, 2, 3, 4)$, ignoring $\epsilon$ and taking $\gamma = g = 1$, $\beta = 0$. LayerNorm: $\mu = 2.5$, $\sigma^2 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25$, $\sigma \approx 1.118$, so the output is $(-1.342, -0.447, 0.447, 1.342)$: mean 0, variance 1. RMSNorm: the mean of squares is $(1 + 4 + 9 + 16)/4 = 7.5$, the root mean square is $\approx 2.739$, so the output is $(0.365, 0.730, 1.095, 1.461)$: same direction as $x$, but of standard size.

**Where to put the normalization?** The original Transformer used **Post-Norm**: $x_{\text{new}} = \text{LayerNorm}(x + F(x))$, normalizing *after* the residual addition. Later work found that **Pre-Norm**, $x_{\text{new}} = x + F(\text{Norm}(x))$, trains more stably, especially in deep models, and is less sensitive to the learning-rate warm-up. The reason is visible in the formula: in Pre-Norm the residual path from input to output contains no normalization at all, so the gradient "highway" is perfectly clean. Almost all modern LLMs use Pre-Norm, plus one extra normalization after the last block.

## The feed-forward network and gated variants

The FFN is a small two-layer network applied to each token's vector *independently* — the same weights for every position:

$$
\text{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2
$$

Here $W_1$ has size $d_{\text{ff}} \times d$ and expands the vector to a larger **hidden dimension** $d_{\text{ff}}$, usually $4d$; $\sigma$ is an activation function (ReLU in the original paper, GELU in BERT and GPT); and $W_2$ of size $d \times d_{\text{ff}}$ projects back to width $d$. Although it looks modest, the FFN holds most of the parameters: with $d_{\text{ff}} = 4d$, its two matrices have $8d^2$ weights, while the four attention matrices have $4d^2$. A useful intuition, supported by interpretability studies, is that FFN layers act like a large key–value memory: rows of $W_1$ detect patterns in the token vector, and the matching columns of $W_2$ write associated information into the residual stream.

**Gated linear units (GLU)** improve the FFN with a multiplicative gate. Shazeer ("GLU Variants Improve Transformer") compared several variants; the one called **SwiGLU** became the standard in LLaMA, PaLM and many others:

$$
\text{SwiGLU}(x) = W_{\text{down}} \big(\, \text{Swish}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x \,\big)
$$

There are now three matrices. $W_{\text{up}} x$ computes candidate features; $\text{Swish}(W_{\text{gate}} x)$, where $\text{Swish}(z) = z \cdot \text{sigmoid}(z)$ (also called SiLU), computes for each feature a soft "how much to let through"; their element-wise product passes through $W_{\text{down}}$. The gate lets the network make one feature depend on another multiplicatively, which plain ReLU or GELU cannot do as easily. To keep the parameter count equal to a standard FFN, the hidden dimension is reduced to about $\tfrac{8}{3} d$: then $3 \cdot d \cdot \tfrac83 d = 8d^2$.

## Putting the block together

Here is a complete Pre-Norm block with RMSNorm and SwiGLU, in the style of modern decoder models. For brevity we use PyTorch's built-in `nn.MultiheadAttention` instead of our class from Chapter 16. Note one trap: in PyTorch's convention a boolean `attn_mask` marks with `True` the positions that are *blocked*, the opposite of the convention we used in Chapter 16.

```python
import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.g = eps, nn.Parameter(torch.ones(d))  # learned gain g
    def forward(self, x):
        return self.g * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x))  # gate * value
class Block(nn.Module):                                   # pre-norm Transformer block
    def __init__(self, d, n_heads, hidden):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = SwiGLU(d, hidden)
    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]  # residual 1
        return x + self.ffn(self.norm2(x))                                  # residual 2

mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)  # PyTorch: True = blocked
block = Block(48, 4, 128)                                 # d = 48, 4 heads, hidden = 8/3 * 48 = 128
print(block(torch.randn(2, 5, 48), mask).shape, sum(p.numel() for p in block.parameters()))
```

The output is `torch.Size([2, 5, 48]) 27936`. The shape is unchanged — a block maps a sequence of vectors to a sequence of vectors of the same size, which is exactly why blocks can be stacked. The parameters split as follows: attention $4 \cdot 48^2 + 4 \cdot 48 = 9408$ (four matrices plus biases), SwiGLU $3 \cdot 48 \cdot 128 = 18432$, and the two RMSNorm gains $2 \cdot 48 = 96$. As predicted, the FFN holds about two thirds of the block.

To summarize the data flow of one block in two lines:

$$
x \leftarrow x + \text{MHA}(\text{Norm}(x)), \qquad x \leftarrow x + \text{FFN}(\text{Norm}(x)).
$$

A full model is: token embeddings (plus positions, or RoPE inside attention) → $N$ blocks → final normalization → a linear layer that turns each vector into vocabulary logits. In Chapter 19 we will see how blocks are arranged into encoders and decoders, and in Chapter 20 we will build and train such a model ourselves.

## Summary

- A Transformer block has two sublayers: multi-head attention (communication between tokens) and a feed-forward network (computation inside each token).
- Residual connections add each sublayer's output to its input; they keep gradients flowing in deep stacks and form the residual stream.
- LayerNorm normalizes each token's vector to zero mean and unit variance; RMSNorm only divides by the root mean square and is cheaper.
- Pre-Norm ($x + F(\text{Norm}(x))$) trains more stably than the original Post-Norm and is used in almost all modern LLMs.
- The FFN expands to a hidden dimension of about $4d$ and holds most parameters; SwiGLU adds a multiplicative gate and uses a hidden size of about $\tfrac83 d$.

## Exercises

1. Explain in your own words the difference between what attention does and what the FFN does inside a block.
2. Apply LayerNorm and RMSNorm to $x = (2, 2, 2, 6)$ by hand (take $\gamma = g = 1$, $\beta = 0$, $\epsilon = 0$).
3. For $d = 4096$, how many parameters does one block have (ignore biases and norms) with a standard FFN ($d_{\text{ff}} = 4d$)? With SwiGLU ($d_{\text{ff}} = \tfrac83 d$)?
4. Change the `Block` class to Post-Norm and train both versions on random data for a few hundred steps with a 24-block stack. Compare how the loss behaves at a high learning rate.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| Transformer block | блок трансформера | 트랜스포머 블록 |
| sublayer | подслой | 하위 층 |
| residual connection | остаточная связь | 잔차 연결 |
| residual stream | остаточный поток | 잔차 스트림 |
| layer normalization | нормализация слоя | 층 정규화 |
| RMSNorm | RMSNorm | RMSNorm |
| root mean square | среднеквадратичное значение | 제곱평균제곱근 |
| element-wise multiplication | поэлементное умножение | 원소별 곱 |
| Pre-Norm / Post-Norm | Pre-Norm / Post-Norm (пред- и постнормализация) | 사전 정규화 / 사후 정규화 |
| feed-forward network | полносвязная сеть прямого распространения | 피드포워드 네트워크 |
| hidden dimension | скрытая размерность | 은닉 차원 |
| activation function | функция активации | 활성화 함수 |
| gated linear unit (GLU) | вентильный линейный блок | 게이트 선형 유닛(GLU) |
| SwiGLU | SwiGLU | SwiGLU |

---
[← Previous: Positional Information: Sinusoidal, Learned, RoPE, ALiBi](17-positional-encoding.md) · [Contents](README.md) · [Next: Encoder, Decoder and Encoder–Decoder Architectures →](19-transformer-architectures.md)
