# Chapter 12. LSTM and GRU

> **Part 3 — Sequences and Attention** · Level: Intermediate
>
> **In this chapter you will learn:** how gates let a network decide what to remember and what to forget · the equations of the LSTM cell and why it eases the vanishing gradient problem · how the GRU simplifies the LSTM and how to choose between them

## A memory with switches

Imagine you are reading a long novel and taking notes on a single sheet of paper. A simple RNN from Chapter 11 behaves like a reader who, after every sentence, rewrites the whole sheet from scratch using the old notes and the new sentence. After a hundred rewrites, details from the first chapter are almost gone — each rewrite blurs them a little more.

A smarter reader does three different things. They **erase** notes that are no longer relevant ("the hero is in Moscow" — but he has now moved to Kazan). They **add** new important facts. And when someone asks a question, they **read out** only the relevant part of the notes. Most of the sheet is simply left as it is from one sentence to the next.

This is the idea of the **Long Short-Term Memory** network (LSTM), introduced by Hochreiter and Schmidhuber (1997). The name sounds strange, but it means "a short-term memory (the hidden state of an RNN) that can last for a long time". The forget mechanism was added a little later by Gers, Schmidhuber and Cummins (2000), and that version is the LSTM used today.

The switches that erase, add and read are called **gates**. A gate is a vector of numbers between 0 and 1, produced by a sigmoid function (Chapter 7). We multiply another vector by a gate element by element: where the gate is close to 1 the value passes through, and where it is close to 0 the value is blocked. Because the sigmoid is smooth, the gates are differentiable and the network learns *when* to open and close them with ordinary gradient descent.

## Inside the LSTM cell

An LSTM keeps two vectors at every time step instead of one:

- the **cell state** $c_t$ — the "sheet of notes", the long-term memory;
- the hidden state $h_t$ — what the cell shows to the outside world and to the next layer.

At step $t$ the cell receives the input $x_t$ and the previous hidden state $h_{t-1}$. From these it computes three gates and one candidate vector. We write $[h_{t-1}, x_t]$ for the concatenation of the two vectors and $\sigma$ for the sigmoid:

$$f_t = \sigma\left(W_f [h_{t-1}, x_t] + b_f\right) \quad \text{(forget gate)}$$

$$i_t = \sigma\left(W_i [h_{t-1}, x_t] + b_i\right) \quad \text{(input gate)}$$

$$o_t = \sigma\left(W_o [h_{t-1}, x_t] + b_o\right) \quad \text{(output gate)}$$

$$\tilde{c}_t = \tanh\left(W_c [h_{t-1}, x_t] + b_c\right) \quad \text{(candidate values)}$$

Then the cell updates its memory and its output:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$h_t = o_t \odot \tanh(c_t)$$

Here $\odot$ denotes **element-wise multiplication** (also called the Hadamard product): $(a \odot b)_k = a_k b_k$. Let us read the equations in words:

- The **forget gate** $f_t$ decides, for every component of the memory, how much of the old value $c_{t-1}$ to keep.
- The **candidate** $\tilde{c}_t$ is new information proposed from the current input, with values in $(-1, 1)$.
- The **input gate** $i_t$ decides how much of this candidate to write into memory.
- The **output gate** $o_t$ decides which parts of the memory to expose as $h_t$.

Each of the four $W$ matrices has size $d_h \times (d_h + d_{in})$. In practice they are stored as one large matrix of size $4d_h \times (d_h + d_{in})$, and the result is split into four parts. The code below does exactly this:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)
d_in, d_h = 3, 4
# One big matrix for all four parts: forget, input, output, candidate
W = rng.normal(0, 0.5, (4 * d_h, d_in + d_h))
b = np.zeros(4 * d_h)
b[:d_h] = 1.0                                # forget-gate bias = 1: "remember by default"

def lstm_step(x, h, c):
    z = W @ np.concatenate([x, h]) + b
    f = sigmoid(z[0*d_h:1*d_h])              # forget gate
    i = sigmoid(z[1*d_h:2*d_h])              # input gate
    o = sigmoid(z[2*d_h:3*d_h])              # output gate
    g = np.tanh(z[3*d_h:4*d_h])              # candidate values
    c = f * c + i * g                        # update the cell state
    h = o * np.tanh(c)                       # new hidden state
    return h, c, f

h, c = np.zeros(d_h), np.zeros(d_h)
for x in rng.normal(size=(5, d_in)):
    h, c, f = lstm_step(x, h, c)
    print("forget gate:", np.round(f, 2), " c:", np.round(c, 2))

m, n = 100, 256                              # parameter counts: input m, hidden n
for name, k in [("RNN", 1), ("GRU", 3), ("LSTM", 4)]:
    print(name, k * (n * (m + n) + n))
```

(In this code the vector is concatenated as $[x_t, h_{t-1}]$; the order does not matter as long as it is the same everywhere.) Note the line `b[:d_h] = 1.0`. Initializing the forget-gate bias to a positive value makes $f_t$ start close to 1, so that a freshly initialized LSTM remembers by default. This simple trick is widely used and usually improves training.

## Why the LSTM remembers longer

Recall from Chapter 11 the cause of vanishing gradients: in a simple RNN, the state is completely recomputed at every step, $h_t = \tanh(W_{hh} h_{t-1} + \dots)$, and the gradient is multiplied by the matrix $W_{hh}$ and the derivative of $\tanh$ at each step back.

Look again at the cell update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The new memory is the old memory, scaled, *plus* something new. If we follow only this direct path from $c_{t-1}$ to $c_t$ (ignoring the smaller indirect paths through the gates), the derivative is very simple:

$$\frac{\partial c_t}{\partial c_{t-1}} \approx \operatorname{diag}(f_t)$$

There is no weight matrix and no $\tanh$ derivative on this path. Across many steps, the gradient is multiplied only by the forget gates: $f_{k+1} \odot f_{k+2} \odot \dots \odot f_t$. When the network wants to keep a piece of information, it learns to set the corresponding components of $f$ close to 1, and then the gradient can flow back over hundreds of steps with little decay. This direct, mostly additive path is sometimes called the "constant error carousel" in the original paper. The same idea — let information flow along an additive path — returns in a different form as the **residual connection** of the Transformer (Chapter 18).

A worked example in one dimension: suppose $c_{t-1} = 2.0$, $f_t = 0.9$, $i_t = 0.2$ and $\tilde{c}_t = -0.5$. Then $c_t = 0.9 \cdot 2.0 + 0.2 \cdot (-0.5) = 1.8 - 0.1 = 1.7$. Most of the old memory survives, and a small correction is added. Over 20 steps with $f = 0.9$ the old content is scaled by $0.9^{20} \approx 0.12$, while with $f = 0.99$ it is scaled by about $0.82$. The network controls its own memory length.

The LSTM does not remove the problem completely — the gates are still sigmoids, and very long dependencies remain hard — but in practice it can learn dependencies over hundreds of steps where a simple RNN fails after a few dozen. Exploding gradients are still possible, so gradient clipping is still used.

## The GRU: a lighter alternative

The **Gated Recurrent Unit** (GRU), proposed by Cho et al. (2014), keeps the main idea of gating but simplifies the design. It has no separate cell state — only $h_t$ — and uses two gates instead of three:

$$z_t = \sigma\left(W_z [h_{t-1}, x_t] + b_z\right) \quad \text{(update gate)}$$

$$r_t = \sigma\left(W_r [h_{t-1}, x_t] + b_r\right) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh\left(W_h [r_t \odot h_{t-1}, x_t] + b_h\right)$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

The **update gate** $z_t$ plays the role of both the forget and the input gate at once: the new state is an interpolation between the old state and the candidate. If $z_t \approx 0$, the old state is copied forward unchanged — the same additive path that helps the LSTM. The **reset gate** $r_t$ decides how much of the old state to use when computing the candidate; when $r_t \approx 0$ the unit "forgets" the past and behaves as if it started reading fresh. (Be careful: some libraries, including PyTorch, swap the roles of $z_t$ and $1 - z_t$ in the last equation. The model is the same; only the convention differs.)

How do they compare in size? A simple RNN has one weight block of size $d_h \times (d_h + d_{in})$ plus a bias; a GRU has three such blocks and an LSTM four. The last lines of the code print these counts for input size 100 and hidden size 256 (with one bias vector per block): 91,392 for the RNN, 274,176 for the GRU and 365,568 for the LSTM.

Which one to choose? Empirical comparisons (for example, Chung et al., 2014; Jozefowicz et al., 2015) found that both clearly beat the simple RNN, while neither wins on every task. The GRU is smaller and slightly faster; the LSTM is a little more expressive and was the more common default. Both can be stacked into several layers and made bidirectional, exactly as in Chapter 11.

## LSTMs in the history of language models

For several years gated RNNs were the standard way to model language. LSTM language models gave much lower perplexity (Chapter 4) than n-gram models, and the first successful neural machine translation systems were built from LSTMs — as we will see in Chapter 13. Contextual word representations such as ELMo (Peters et al., 2018) came from a deep bidirectional LSTM.

Yet gating solved only the memory problem. Two others from Chapter 11 remain. Computation is still sequential: $h_t$ needs $h_{t-1}$, so long sequences cannot be processed in parallel. And the whole past is still squeezed into one fixed-size vector. The next two chapters show how attention removes the second limitation, and Chapter 15 shows how self-attention removes the first.

## Summary

- Gates are vectors of values in $(0, 1)$, produced by a sigmoid, that control information flow by element-wise multiplication.
- The LSTM keeps a cell state $c_t$ and uses forget, input and output gates: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, $h_t = o_t \odot \tanh(c_t)$.
- The additive cell update gives the gradient a path without weight matrices, so information can survive for hundreds of steps.
- Initializing the forget-gate bias to a positive value makes the LSTM remember by default.
- The GRU merges the cell and hidden state and uses update and reset gates; it has about three quarters of the LSTM's parameters.
- Gated RNNs fix memory but remain sequential and still compress the past into a fixed-size vector.

## Exercises

1. Explain in your own words the role of each LSTM gate, using the analogy of a reader taking notes. What happens if the forget gate is always 0? Always 1?
2. In one dimension, let $c_0 = 1$, and at every step $f_t = 0.95$ and $i_t = 0$. What is $c_{50}$? How large must $f$ be so that $c_{100} \ge 0.5$?
3. Compute the number of parameters of an LSTM layer with input size $d_{in} = 300$ and hidden size $d_h = 512$, using one bias vector per gate. Do the same for a GRU.
4. Write a function `gru_step(x, h)` in NumPy following the equations above (use the code of this chapter as a template), and run it over a sequence of five random vectors.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| long short-term memory (LSTM) | долгая краткосрочная память | 장단기 메모리 |
| gate | вентиль | 게이트 |
| cell state | состояние ячейки | 셀 상태 |
| forget gate | вентиль забывания | 망각 게이트 |
| input gate | входной вентиль | 입력 게이트 |
| output gate | выходной вентиль | 출력 게이트 |
| candidate values | значения-кандидаты | 후보 값 |
| element-wise multiplication | поэлементное умножение | 원소별 곱셈 |
| gated recurrent unit (GRU) | управляемый рекуррентный блок | 게이트 순환 유닛 |
| update gate | вентиль обновления | 업데이트 게이트 |
| reset gate | вентиль сброса | 리셋 게이트 |
| sigmoid | сигмоида | 시그모이드 |
| residual connection | остаточная связь | 잔차 연결 |

---
[← Previous: Recurrent Neural Networks](11-recurrent-networks.md) · [Contents](README.md) · [Next: Sequence-to-Sequence and the Encoder–Decoder →](13-sequence-to-sequence.md)
